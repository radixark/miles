"""Bridge / LoRA model setup helpers.

Extracted from ``model.py`` to keep the main training module focused on
forward / backward / optimizer logic.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from dataclasses import dataclass

from megatron.core.utils import get_attr_wrapped_model

from miles.utils.hf_config import load_hf_config
from miles.utils.multi_lora import is_multi_lora_enabled, targets_expert_leaves

from .lora_utils import convert_target_modules_to_hf, patch_param_grad_buffer_for_colocate_mode_lora

logger = logging.getLogger(__name__)


@dataclass
class _BridgeWrapperConfig:
    """Configuration for Megatron-Bridge module wrapping."""

    is_value_model: bool = False
    wrap_with_ddp: bool = True
    use_distributed_optimizer: bool = True


def _ensure_model_list(model):
    return model if isinstance(model, list) else [model]


def _ignore_packed_sequence_boundaries(module) -> None:
    """Let one-sample GDN microbatches use the THD tensor layout.

    Megatron's GDN rejects ``PackedSeqParams`` even when the packed row contains
    only one real sample followed by masked padding. In that exact case the
    boundaries do not affect any trained token, so the recurrent layer can run
    its ordinary dense path while full-attention layers keep the THD metadata.
    """
    if getattr(module, "_miles_ignores_packed_sequence_boundaries", False):
        return

    original_forward = module.forward

    def forward(*forward_args, **forward_kwargs):
        forward_kwargs["packed_seq_params"] = None
        return original_forward(*forward_args, **forward_kwargs)

    module.forward = forward
    module._miles_ignores_packed_sequence_boundaries = True


def _enable_single_sample_gdn_thd(model, args: Namespace) -> int:
    """Enable the safe GDN fallback only for fixed, one-sample THD batches."""
    if not (
        getattr(args, "qkv_format", "thd") == "thd"
        and getattr(args, "micro_batch_size", None) == 1
        and not getattr(args, "use_dynamic_batch_size", False)
        and getattr(args, "context_parallel_size", 1) == 1
    ):
        return 0

    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    patched = 0
    for model_chunk in _ensure_model_list(model):
        for module in model_chunk.modules():
            if isinstance(module, GatedDeltaNet):
                _ignore_packed_sequence_boundaries(module)
                patched += 1
    if patched:
        logger.info(
            "Enabled the single-sample THD fallback for %d GatedDeltaNet layers; "
            "full-attention layers retain packed sequence boundaries.",
            patched,
        )
    return patched


def _make_value_model_hook(hidden_size: int, sequence_parallel: bool):
    """Create a pre-wrap hook that replaces the output layer with a value head."""
    from megatron.core import parallel_state

    from .model_provider import LinearForLastLayer

    def hook(model):
        model_post_process = []
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            for i in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
                model_post_process.append(parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i))
        else:
            model_post_process.append(parallel_state.is_pipeline_last_stage())

        model_list = _ensure_model_list(model)
        assert len(model_post_process) == len(model_list), "Model list length and post process list length must match."

        for index, model_chunk in enumerate(model_list):
            if not model_post_process[index]:
                continue
            model_chunk.output_layer = LinearForLastLayer(
                input_size=hidden_size,
                output_size=1,
                sequence_parallel=sequence_parallel,
            )

    return hook


def _get_model_config_from_wrapped(model):
    return get_attr_wrapped_model(model, "config", allow_none=False)


def _validate_multi_lora_moe_support(args: Namespace, provider) -> None:
    """Reject MoE configs the multi-slot grouped-expert adapter cannot serve (checked
    post-finalize because they depend on the resolved provider, not the CLI)."""
    if not getattr(provider, "num_moe_experts", None):
        return
    if not targets_expert_leaves(args.target_modules):
        logger.info("[multilora] MoE model with no expert leaves in --target-modules; experts stay frozen")
        return

    # Checked on the provider: --expert-tensor-parallel-size stays None until Megatron resolves it.
    expert_tp = getattr(provider, "expert_tensor_parallel_size", 1) or 1
    assert expert_tp == 1, (
        f"Multi-LoRA on MoE experts requires expert_tensor_parallel_size=1 (resolved to "
        f"{expert_tp}); set --expert-tensor-parallel-size 1."
    )
    assert getattr(provider, "moe_grouped_gemm", False), (
        "Multi-LoRA on MoE experts requires moe_grouped_gemm=True (SequentialMLP expert "
        "linears are skipped, so the experts would train no adapter)."
    )
    assert not getattr(provider, "fp8", None) and not getattr(provider, "fp4", None), (
        "Multi-LoRA on MoE experts does not support fp8/fp4 experts (quantization padding "
        "desynchronizes the dispatched token order)."
    )
    # sglang only wraps a fused MoE layer when both expert projections are targeted.
    served = set(convert_target_modules_to_hf(list(args.target_modules)))
    expert_pair = {"gate_proj", "up_proj", "down_proj"}
    if served & expert_pair:
        assert expert_pair <= served, (
            f"Multi-LoRA on MoE experts requires all of {sorted(expert_pair)} in "
            f"--target-modules (got {sorted(served & expert_pair)}); a one-sided expert "
            f"target is dropped at rollout time."
        )
    assert not getattr(
        provider, "moe_pad_expert_input_to_capacity", False
    ), "Multi-LoRA on MoE experts does not support --moe-pad-expert-input-to-capacity."
    assert not getattr(
        provider, "moe_permute_fusion", False
    ), "Multi-LoRA on MoE experts requires moe_permute_fusion=False."


def _setup_lora_model_via_bridge(args: Namespace) -> list:
    """Build Megatron model with LoRA using Megatron-Bridge.

    This handles:
    1. Creating the Bridge and Provider
    2. Creating and registering the LoRA pre-wrap hook
    3. Registering value-model hooks if needed
    4. Building the DDP-wrapped model

    Args:
        args: Training arguments.

    Returns:
        List of DDP-wrapped model chunks with LoRA applied.
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.config import DistributedDataParallelConfig

    hf_config = load_hf_config(args.hf_checkpoint)
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)

    # LoRA models use this separate Bridge construction path, so apply the
    # same explicit runtime overrides as the non-LoRA provider before the
    # LoRA-specific settings below are finalized.
    from .model_provider import _apply_bridge_runtime_config

    _apply_bridge_runtime_config(provider, args)

    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    provider.expert_model_parallel_size = args.expert_model_parallel_size
    provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    provider.sequence_parallel = args.sequence_parallel
    provider.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
    provider.context_parallel_size = args.context_parallel_size
    provider.gradient_accumulation_fusion = args.gradient_accumulation_fusion
    provider.recompute_granularity = args.recompute_granularity
    provider.recompute_method = args.recompute_method
    provider.recompute_num_layers = args.recompute_num_layers
    provider.recompute_modules = args.recompute_modules
    provider.distribute_saved_activations = args.distribute_saved_activations
    provider.attention_backend = args.attention_backend
    provider.variable_seq_lengths = True
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_router_load_balancing_type = "none"
    if is_multi_lora_enabled(args) and targets_expert_leaves(args.target_modules):
        # Expert adapters cannot replay the fused permute's row_id_map, and most bridge
        # MoE providers default the fusion on — so turn it off rather than refuse to build.
        if getattr(provider, "moe_permute_fusion", False):
            logger.info(
                "[multilora] disabling moe_permute_fusion: expert adapters replay the "
                "dispatcher's permutation, which the fused kernel does not expose"
            )
        provider.moe_permute_fusion = False
    if getattr(args, "decoder_first_pipeline_num_layers", None) is not None:
        provider.num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
    if getattr(args, "decoder_last_pipeline_num_layers", None) is not None:
        provider.num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers
    if hasattr(provider, "dsa_attention_backend"):
        provider.dsa_attention_backend = getattr(args, "dsa_attention_backend", "megatron")
    provider.finalize()

    if is_multi_lora_enabled(args):
        _validate_multi_lora_moe_support(args, provider)

        from miles.backends.megatron_utils.tinker_backend.model import create_multi_lora_instance

        lora = create_multi_lora_instance(args)
    else:
        from .lora_utils import create_lora_instance

        lora = create_lora_instance(args)

    def apply_lora_hook(model_chunks):
        transformed = lora(model_chunks, training=True)
        lora.set_params_to_save(transformed)
        return transformed

    provider.register_pre_wrap_hook(apply_lora_hook)

    is_value_model = (
        "ForTokenClassification" in hf_config.architectures[0]
        or "ForSequenceClassification" in hf_config.architectures[0]
    )
    if is_value_model:
        hidden_size = hf_config.text_config.hidden_size if hasattr(hf_config, "text_config") else hf_config.hidden_size
        provider.register_pre_wrap_hook(_make_value_model_hook(hidden_size, provider.sequence_parallel))

    use_distributed_optimizer = "muon" not in (args.optimizer or "").lower()
    if is_multi_lora_enabled(args):
        # Per-slot LayerWise optimizers: plain DDP all-reduce keeps full grads on
        # every rank (whole-param sharding + retained-gradient idempotency).
        use_distributed_optimizer = False
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=use_distributed_optimizer,
        grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,
    )
    ddp_config.finalize()

    if args.offload_train:
        patch_param_grad_buffer_for_colocate_mode_lora()

    model = provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config)
    _enable_single_sample_gdn_thd(model, args)
    return model
