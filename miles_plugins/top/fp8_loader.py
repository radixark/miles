"""Install a block-FP8 HF checkpoint into Miles' TOP model as trainable FP8 masters.

GLM-5 (glm_moe_dsa), DeepSeek-V4 and DeepSeek-V4.1 FP8 checkpoints load through GLMCheckpoint /
DSV4Checkpoint / DSV41Checkpoint inside the model provider, before Float16Module and DDP,
so the masters are the parameters DDP buckets and the optimizer steps. The install reads
the HF checkpoint directly, so an HF ``--load`` is redundant and skipped. The processed FP8
caches are rebuilt from the masters on every weight event (weight_events): optimizer step,
checkpoint load, weight restore and weight sync.
"""

from __future__ import annotations

import torch

_FAMILIES = ("glm_moe_dsa", "deepseek_v4", "deepseek_v41")


def wanted(args) -> bool:
    """TOP contract, a GLM-5, DeepSeek-V4 or DeepSeek-V4.1 family, and a block-FP8 checkpoint."""
    if args.true_on_policy_contract is None:
        return False
    from miles_plugins.top.binding import hf_model_type
    from miles_plugins.top.fp8_config import checkpoint_is_block_fp8

    return hf_model_type(args) in _FAMILIES and checkpoint_is_block_fp8(args.hf_checkpoint)


def refuse_unsupported(args) -> None:
    """Refuse the trainer paths the FP8 masters do not cover; each would otherwise run silently wrong."""
    from miles.utils.lora.utils import is_lora_enabled

    if args.backend.overlap_param_gather:
        raise NotImplementedError(
            "[top] FP8 masters read expert and kv_b weights outside their modules' forward, so the param all-gather "
            "of --overlap-param-gather is never waited on before the FP8 caches rebuild; drop --overlap-param-gather"
        )
    if is_lora_enabled(args):
        raise NotImplementedError("[top] LoRA does not wrap the TOP FP8 linears; FP8 masters train the full weights")
    if args.save_hf is not None or (args.eval_uses_snapshots and args.eval_hf_dir is not None):
        raise NotImplementedError(
            "[top] --save-hf and snapshot eval would export the weight-sync form under TOP (packed UE8M0 scales "
            "beside the checkpoint's FP32-scale config), which HF and stock SGLang cannot load; save Megatron "
            "checkpoints instead"
        )


def install(args, model) -> None:
    """Replace the provider-built weights of ``model`` (one GPTModel chunk) with the HF checkpoint's."""
    from megatron.core import parallel_state as ps

    from miles_plugins.top.binding import hf_model_type

    refuse_unsupported(args)
    if getattr(model, "vp_stage", None) not in (
        None,
        0,
    ):  # config-access-exempt: module variants differ in whether they define vp_stage
        raise NotImplementedError("[top] trainable FP8 loader does not support virtual pipeline stages")
    ep_rank, ep_size = ps.get_expert_model_parallel_rank(), ps.get_expert_model_parallel_world_size()
    # Cast to BF16 on CUDA before the install, so the cast leaves the dtypes the install sets alone; what stays FP32
    # afterwards is meant to.
    model.cuda(torch.cuda.current_device()).bfloat16()
    decoder = model.decoder
    if hf_model_type(args) == "glm_moe_dsa":
        from miles_plugins.top.glm_checkpoint import GLMCheckpoint

        ck = GLMCheckpoint(args.hf_checkpoint)
        for layer in decoder.layers:
            ck.install_layer(layer, layer.layer_number - 1, ep_rank, ep_size)
        names = ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
    elif hf_model_type(args) == "deepseek_v41":
        from miles_plugins.top.dsv41_checkpoint import DSV41Checkpoint
        from miles_plugins.top.dsv41_spec import reset_block_state

        ck = DSV41Checkpoint(args.hf_checkpoint)
        fp8_config = ck.fp8_config()
        for layer in decoder.layers:
            ck.install_layer(layer, layer.layer_number - 1, ep_rank, ep_size, fp8_config)
        model.register_forward_pre_hook(reset_block_state, with_kwargs=True)
        from miles_plugins.top import dsv41_pipeline

        dsv41_pipeline.install(model)
        names = ("embed.weight", "norm.weight", "head.weight")
    else:
        from miles_plugins.top.dsv4_checkpoint import DSV4Checkpoint

        ck = DSV4Checkpoint(args.hf_checkpoint)
        fp8_config = ck.fp8_config()
        for layer in decoder.layers:
            ck.install_layer(layer, layer.layer_number - 1, ep_rank, ep_size, fp8_config)
        if decoder.hc_head_contraction is not None:
            ck.install_head(decoder.hc_head_contraction)
        names = ("embed.weight", "norm.weight", "head.weight")
    embed, norm, head = names
    # Layers carry global numbers; the embedding lives on the first pipeline stage, norm and head on the last.
    if model.pre_process:
        _copy_rows(model.embedding.word_embeddings.weight, ck.tensor(embed))
    if decoder.final_layernorm is not None:
        decoder.final_layernorm.weight.data.copy_(ck.tensor(norm))
    if model.post_process:
        _copy_rows(model.output_layer.weight, ck.tensor(head))
    for tensor in [*model.parameters(), *model.buffers()]:
        if tensor.is_floating_point() and tensor.dtype == torch.float32:
            tensor.keep_in_fp32 = True  # Float16Module leaves it alone
    from miles.backends.megatron_utils import weight_events

    weight_events.on_weights_changed(rebuild_caches)
    # Stale caches are never read again; drop them before the offload backs them up, rebuild before training.
    weight_events.on_offload(drop_stale_caches)
    weight_events.on_train_start(materialize_caches)
    weight_events.mark_hf_checkpoint_installed(args.hf_checkpoint)


def _copy_rows(param, value):
    """Vocab-padded embedding/head at TP1: checkpoint rows first, padding rows zero."""
    if param.shape[1:] != value.shape[1:] or param.shape[0] < value.shape[0]:
        raise ValueError(f"[top] checkpoint {tuple(value.shape)} does not fit parameter {tuple(param.shape)}")
    param.data.zero_()
    param.data[: value.shape[0]].copy_(value)


def rebuild_caches() -> None:
    """Rebuild the processed FP8 caches from the masters.

    After a load or weight restore the masters changed. After a weight sync serving holds the sync
    quantizer's output, which for non-power-of-two checkpoint scales (GLM) is not the checkpoint's
    bytes; Miles syncs before the first rollout, so even step 0 must read the synced form."""
    from miles_plugins.top.fp8_masters import advance_weights_epoch

    advance_weights_epoch()


def drop_stale_caches() -> int:
    """Free the caches the last weight event made stale (before the train offload); returns the bytes freed."""
    from miles_plugins.top.fp8_masters import drop_stale_caches as drop

    return drop()


def materialize_caches() -> None:
    """Rebuild stale caches from the restored masters ahead of the first forward."""
    from miles_plugins.top.fp8_masters import materialize_caches as materialize

    materialize()
