"""Single public entry point and lifecycle orchestration for native LoRA."""

from __future__ import annotations

import logging

import torch.nn as nn

from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.hf_adapter import (
    export_lora_hf_named,
    load_lora_adapter_hf,
    mbridge_cross_check,
    resolve_hf_naming,
)
from miles_plugins.lora.registry import resolve_model_spec
from miles_plugins.lora.sglang_adapter import export_lora_sglang_named
from miles_plugins.lora.spec.base import AttachContext, AttentionFamily

logger = logging.getLogger(__name__)


def _require_grad_on_first_activation(model) -> nn.Module | None:
    """Make a frozen embedding output require grad so recomputation enters each block."""
    embedding = getattr(model, "embedding", None)
    if embedding is None:
        return None

    def hook(_module, _inputs, output):
        return output if output.requires_grad else output.requires_grad_(True)

    embedding.register_forward_hook(hook)
    return embedding


def _resolve_attach_context(args, transformer_config, arch_spec) -> tuple[LoRAConfig, AttachContext]:
    from megatron.core import parallel_state as ps

    requested_config = LoRAConfig.from_args(args)
    lora_config = arch_spec.normalize_config(requested_config)
    removed_targets = requested_config.target_modules - lora_config.target_modules
    if removed_targets:
        logger.info(
            "[lora-native] %s spec removed architecture-neutral compatibility targets: %s",
            arch_spec.name,
            sorted(removed_targets),
        )
    layer_prefix, shared_expert = resolve_hf_naming(getattr(args, "hf_checkpoint", None))
    context = AttachContext(
        lora=lora_config,
        transformer_config=transformer_config,
        tp_size=ps.get_tensor_model_parallel_world_size(),
        tp_rank=ps.get_tensor_model_parallel_rank(),
        layer_prefix=layer_prefix,
        shared_expert=shared_expert,
    )
    return lora_config, context


def _assert_supported_run(args, context: AttachContext) -> None:
    """Reject runtime interactions that have not been validated for native LoRA."""
    assert not getattr(args, "overlap_param_gather", False), (
        "native LoRA does not yet support --overlap-param-gather: adapters now have a real module call "
        "path, but their sibling-module attachment has not been validated against MCore's bucket "
        "prefetch ordering. Drop the flag, or use --megatron-to-hf-mode bridge."
    )
    assert not getattr(args, "moe_shared_expert_overlap", False), (
        "native LoRA does not support --moe-shared-expert-overlap: the dispatcher owns the shared-expert "
        "communication, so the adapter's gather/reduce no longer matches the module's effective parallel "
        "mode. Drop the flag, or use --megatron-to-hf-mode bridge."
    )
    assert not getattr(args, "overlap_grad_reduce", False), (
        "native LoRA does not support --overlap-grad-reduce: replicated adapter gradients need a "
        "tensor-parallel sum (reduce_marked_lora_grads) over the same buffer MCore's per-bucket "
        "data-parallel reduce-scatter writes, and with overlap those collectives are already in flight "
        "on another stream when the sum runs, so adapter gradients come out nondeterministic. The sum "
        "cannot simply move after finish_grad_sync either: with --use-distributed-optimizer the "
        "reduce-scatter leaves only this rank's shard of main_grad valid. Drop the flag. Bridge mode is "
        "not an escape hatch here — its DDP config does not forward the flag either "
        "(miles.backends.megatron_utils.bridge_lora_helpers) — so the flag is inert under LoRA today; "
        "removing this restriction means making adapters real MCore parallel linears (see "
        "miles_plugins/lora/distributed.py's roadmap note)."
    )
    if getattr(args, "colocate", False) and context.targets:
        assert getattr(args, "enable_weights_backuper", True), (
            "native LoRA under --colocate needs the weights backuper: adapter pages are memory-saver-paused "
            "while export runs. Keep the backuper enabled, or drop --colocate."
        )


def _validate_plain_gqa_chunk(layers, arch_spec, context: AttachContext) -> None:
    """Reject a registry/structure mismatch before attaching or freezing anything."""
    attention_targets = context.targets.intersection(arch_spec.attention.supported_targets)
    if arch_spec.name != AttentionFamily.GQA or not attention_targets:
        return
    missing = [
        layer.layer_number - 1 for layer in layers if not hasattr(getattr(layer, "self_attention", None), "linear_qkv")
    ]
    assert not missing, (
        "native LoRA's plain-GQA spec expected self_attention.linear_qkv in "
        f"layers {missing[:8]} for targets {sorted(attention_targets)}, but the projection is missing. "
        "Register the model under a hybrid/mixer-aware spec, use --megatron-to-hf-mode bridge, "
        "or point --lora-provider-path at a model-specific provider."
    )


def apply_native_lora(model, args):
    """Attach native LoRA to one model chunk before Float16Module/DDP wrapping."""
    transformer_config = model.config
    model_type, arch_spec = resolve_model_spec(args, transformer_config)
    lora_config, context = _resolve_attach_context(args, transformer_config, arch_spec)
    arch_spec.validate(context)
    _assert_supported_run(args, context)
    mbridge_cross_check(model_type, context.layer_prefix)
    layers = model.decoder.layers
    _validate_plain_gqa_chunk(layers, arch_spec, context)

    for parameter in model.parameters():
        parameter.requires_grad = False
    hooked_embedding = _require_grad_on_first_activation(model)

    attention_prefix = getattr(getattr(arch_spec.attention, "layout", None), "hf_block_prefix", None) or "self_attn."
    mlp_prefix = getattr(getattr(arch_spec.mlp, "layout", None), "hf_block_prefix", None) or "mlp."
    moe_attach = getattr(arch_spec.moe, "attach", None)

    wrapped = 0
    mixer_only_layers = []
    moe_skipped_targets: set[str] = set()
    for layer in layers:
        layer_index = layer.layer_number - 1
        hf_layer = f"{context.layer_prefix}{layer_index}."
        attention = getattr(layer, "self_attention", None)
        if attention is not None:
            attached = arch_spec.attention.attach(attention, hf_layer + attention_prefix, context)
            wrapped += attached
            if (
                attached == 0
                and not hasattr(attention, "linear_qkv")
                and arch_spec.model_family == AttentionFamily.GQA
            ):
                mixer_only_layers.append(layer_index)

        mlp = layer.mlp
        moe_skipped_targets.update(arch_spec.moe.validate_layer(mlp, context))
        if hasattr(mlp, "linear_fc1"):
            assert getattr(mlp.config, "gated_linear_unit", True), "native LoRA assumes a gated (SwiGLU) MLP"
            wrapped += arch_spec.mlp.attach(mlp, hf_layer + mlp_prefix, context)
        shared = getattr(mlp, "shared_experts", None)
        if shared is not None and hasattr(shared, "linear_fc1"):
            wrapped += arch_spec.mlp.attach(shared, hf_layer + context.shared_expert, context)
        if moe_attach is not None:
            wrapped += moe_attach(mlp, hf_layer, context)

    if arch_spec.lm_head is not None:
        wrapped += arch_spec.lm_head.attach(model, args, context)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "[lora-native] arch=%s spec=%s rank=%d alpha=%s scale=%.3f dropout=%s targets=%s | "
        "%d modules wrapped, trainable %s / %s params (%.4f%%), input-grad hook=%s",
        model_type or "structural-test-fallback",
        arch_spec.name,
        lora_config.rank,
        lora_config.alpha,
        lora_config.scale,
        lora_config.dropout,
        sorted(lora_config.target_modules),
        wrapped,
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
        hooked_embedding is not None,
    )
    if moe_skipped_targets:
        logger.info(
            "[lora-native] all-linear MLP targets %s skipped on MoE layers without an attachable "
            "shared expert; routed/grouped expert LoRA needs --megatron-to-hf-mode bridge or a "
            "model-specific --lora-provider-path.",
            sorted(moe_skipped_targets),
        )
    if mixer_only_layers:
        shown = f"{mixer_only_layers[:4]}{'...' if len(mixer_only_layers) > 4 else ''}"
        logger.info(
            "[lora-native] %d of %d layers use a GDN/linear-attention mixer and carry no native attention "
            "adapter: %s. GDN projections require a future architecture spec.",
            len(mixer_only_layers),
            len(model.decoder.layers),
            shown,
        )
    pp_size = int(getattr(args, "pipeline_model_parallel_size", 1) or 1)
    vpp_size = int(getattr(args, "virtual_pipeline_model_parallel_size", 1) or 1)
    partitioned_empty_stage = (pp_size > 1 or vpp_size > 1) and not bool(layers)
    mixer_only_attention_chunk = (
        arch_spec.allows_mixer_only_adapter_chunks
        and bool(layers)
        and len(mixer_only_layers) == len(layers)
        and bool(context.targets)
        and context.targets <= arch_spec.attention.supported_targets
    )
    legal_empty_chunk = partitioned_empty_stage or mixer_only_attention_chunk
    if wrapped == 0 and legal_empty_chunk:
        logger.info(
            "[lora-native] this PP/VPP or hybrid-mixer chunk carries no adapter for targets %s; "
            "other model chunks may carry the requested projections",
            sorted(context.targets),
        )
    else:
        assert wrapped > 0, (
            f"native LoRA matched no modules for --target-modules {sorted(context.targets)}; "
            f"the {arch_spec.name} spec supports {sorted(arch_spec.supported_targets)}"
        )
    model._miles_native_lora_provider = True
    return model


def wrap_model_provider_with_lora(provider_func, args):
    """Wrap a Miles model provider so every model chunk gets native LoRA."""

    def wrapped(*provider_args, **provider_kwargs):
        return apply_native_lora(provider_func(*provider_args, **provider_kwargs), args)

    return wrapped


__all__ = [
    "apply_native_lora",
    "export_lora_hf_named",
    "export_lora_sglang_named",
    "load_lora_adapter_hf",
    "wrap_model_provider_with_lora",
]
