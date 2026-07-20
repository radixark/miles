"""Build, shard, and optimize a torchtitan model — the training-side core.

Calls ``model_spec.model.update_from_config`` directly, the same call torchtitan's own
RL ``PolicyTrainer._build_model`` makes (experiments/rl/actors/trainer.py) — passing a
duck-typed config object (only ``.parallelism`` is required; see
``models/common/decoder.py``'s ``Decoder.Config.update_from_config``, which does
``isinstance(config, Trainer.Config)`` to decide whether to also run the
Trainer-specific RoPE/debug-flag propagation, and skips that branch for any other
config-like object). We are not a ``Trainer.Config``, so — exactly like the RL
trainer — we redo the RoPE bounds-check-and-resize ourselves right after the call.
This gets the model-family sharding config (``set_qwen3_sharding_config`` etc.) AND
every other assertion ``update_from_config`` performs (TP-divides-heads, weight-tying
vs PP, MoE-requires-EP, ``maybe_update_minimal_async_ep_config``) for free, instead of
re-deriving a subset of them by hand.

fp32 master weights + bf16 mixed-precision forward: the trainer's forward runs in the
same bf16 the rollout engine serves — torchtitan's own generator-parity recipe.
"""

import dataclasses
import logging
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)


def _apply_update_from_config(spec_model_config, *, parallelism, seq_len: int) -> None:
    from torchtitan.models.common.attention import FlexAttention, VarlenAttention

    spec_model_config.update_from_config(config=SimpleNamespace(parallelism=parallelism))

    # first_attention skips GDN-only layers (e.g. qwen3_5's non-full layers have
    # attention=None) to find the first layer that actually carries an attention config.
    first_attention = spec_model_config.first_attention
    assert first_attention is not None and isinstance(
        first_attention.inner_attention,
        (VarlenAttention.Config, FlexAttention.Config),
    ), "Only varlen and flex attention backends are allowed (matches titan RL trainer's own assert)."

    max_seq_len = spec_model_config.max_seq_len
    if seq_len > max_seq_len:
        raise ValueError(
            f"Training sequence length {seq_len} exceeds attention RoPE maximum "
            f"supported sequence length {max_seq_len}."
        )
    for layer_cfg in spec_model_config.layers:
        attn = getattr(layer_cfg, "attention", None)
        if attn is not None:
            attn.rope = dataclasses.replace(attn.rope, max_seq_len=seq_len)


def build_and_load_model(
    spec,
    hf_checkpoint: str,
    *,
    parallel_dims,
    seq_len: int,
    args,
    device: str = "cuda",
):
    """torchtitan build sequence, matching its own RL PolicyTrainer._build_model
    exactly: update_from_config -> meta-build -> parallelize_fn -> to_empty ->
    init_weights. HF-checkpoint loading is done separately by checkpoint.py's
    CheckpointManager wrapper (torchtitan's own checkpoint stack), not here.
    """
    from torchtitan.config import TORCH_DTYPE_MAP, CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.tools.utils import set_default_dtype

    tp_size = args.tt_tensor_parallel_size
    ep_size = args.tt_expert_parallel_size

    parallelism = ParallelismConfig(
        data_parallel_shard_degree=parallel_dims.dp_shard,
        tensor_parallel_degree=tp_size,
        expert_parallel_degree=ep_size,
    )
    training = TrainingConfig(
        seq_len=seq_len,
        dtype="float32",  # fp32 master weights
        mixed_precision_param="bfloat16",  # bf16 forward, matches sglang's serving dtype
        mixed_precision_reduce="float32",
    )

    _apply_update_from_config(spec.model, parallelism=parallelism, seq_len=seq_len)

    with torch.device("meta"):
        with set_default_dtype(TORCH_DTYPE_MAP[training.dtype]):
            model = spec.model.build()

    if getattr(model, "vision_encoder", None) is not None:
        # Text-only RL training (qwen3_5): prune before to_empty/init_weights so the
        # vision tower is never materialized. torchtitan's own parallelize/pipeline
        # code already null-guards `model.vision_encoder is not None` everywhere
        # (the sanctioned "PP-prune" pattern), and the state_dict_adapter only emits
        # vision_encoder.* keys for tensors actually present in model.state_dict().
        model.vision_encoder = None

    model = spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=CompileConfig(enable=args.tt_compile),
        ac_config=None if args.tt_ac_mode == "none" else _build_ac_config(args.tt_ac_mode),
        dump_folder="/tmp/torchtitan_dump",
    )
    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights(buffer_device=None)

    adapter = spec.state_dict_adapter(spec.model, hf_checkpoint)
    return model, adapter


def _build_ac_config(mode: str):
    from torchtitan.distributed.activation_checkpoint import FullAC, MemoryBudgetAC, SelectiveAC

    if mode == "selective":
        return SelectiveAC.Config()
    if mode == "full":
        return FullAC.Config()
    if mode == "memory-budget":
        return MemoryBudgetAC.Config()
    raise ValueError(f"unknown tt_ac_mode={mode!r}")


def _load_hf_checkpoint(model, adapter, hf_checkpoint: str) -> None:
    """Stream HF safetensors directly into the (possibly sharded) titan model via DCP —
    no rank-0 broadcast, no full materialization; only keys the model emits are ever
    requested (so mtp.*/visual.* extras on a checkpoint are tolerated by construction)."""
    import torch.distributed.checkpoint as dcp

    hf_sd = adapter.to_hf(model.state_dict())
    dcp.load(hf_sd, storage_reader=adapter.get_hf_storage_reader(hf_checkpoint))
    model.load_state_dict(adapter.from_hf(hf_sd))


def build_optimizer_and_lr_scheduler(model, spec, args, parallel_dims, *, training_steps: int):
    from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig, register_moe_load_balancing_hook
    from torchtitan.components.lr_scheduler import LRSchedulersContainer

    opt_config = OptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": args.lr,
                    "betas": (args.adam_beta1, args.adam_beta2),
                    "eps": args.adam_eps,
                    "weight_decay": args.weight_decay,
                },
            )
        ],
    )
    optimizers = opt_config.build(model_parts=[model])
    if spec.post_optimizer_build_fn is not None and spec.post_optimizer_build_fn is not register_moe_load_balancing_hook:
        spec.post_optimizer_build_fn(optimizers, [model], parallel_dims)
    elif spec.post_optimizer_build_fn is register_moe_load_balancing_hook:
        register_moe_load_balancing_hook(optimizers, [model], parallel_dims)

    lr_config = LRSchedulersContainer.Config(
        warmup_steps=args.lr_warmup_iters,
        total_steps=training_steps,
    )
    lr_schedulers = lr_config.build(optimizers=optimizers, training_steps=training_steps)
    return optimizers, lr_schedulers


def clip_grad_norm(model, max_norm: float, parallel_dims) -> torch.Tensor:
    from torchtitan.distributed import utils as dist_utils

    return dist_utils.clip_grad_norm_(
        list(model.parameters()),
        max_norm,
        pp_mesh=None,
        ep_enabled=parallel_dims.ep_enabled,
    )
