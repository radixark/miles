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
from miles_plugins.lora.registry import resolve_checkpoint_spec
from miles_plugins.lora.sglang_adapter import export_lora_sglang_named
from miles_plugins.lora.spec.base import AttachContext, LoRAArchSpec

logger = logging.getLogger(__name__)


def apply_native_lora(model, args):
    """Attach native LoRA to one model chunk before Float16Module/DDP wrapping."""
    model_type, arch_spec = resolve_checkpoint_spec(args.hf_checkpoint)
    context = _resolve_attach_context(args, model.config, arch_spec)
    arch_spec.validate(context)
    _assert_supported_run(args)
    mbridge_cross_check(model_type, context.layer_prefix)

    for parameter in model.parameters():
        parameter.requires_grad = False
    hooked_embedding = _require_grad_on_first_activation(model)

    wrapped = sum(_attach_layer(layer, arch_spec, context) for layer in model.decoder.layers)
    if arch_spec.lm_head is not None:
        wrapped += arch_spec.lm_head.attach(model, args, context)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "[lora-native] arch=%s spec=%s rank=%d alpha=%s dropout=%s targets=%s | %d modules wrapped, "
        "trainable %s / %s params (%.4f%%), input-grad hook=%s",
        model_type,
        arch_spec.name,
        context.rank,
        context.lora.alpha,
        context.dropout,
        "all" if context.lora.target_modules is None else list(context.lora.target_modules),
        wrapped,
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
        hooked_embedding is not None,
    )
    return model


def wrap_model_provider_with_lora(provider_func, args):
    """Wrap a Miles model provider so every model chunk gets native LoRA."""

    def wrapped(*provider_args, **provider_kwargs):
        return apply_native_lora(provider_func(*provider_args, **provider_kwargs), args)

    return wrapped


def _resolve_attach_context(args, transformer_config, arch_spec: LoRAArchSpec) -> AttachContext:
    from megatron.core import parallel_state as ps

    layer_prefix, shared_expert = resolve_hf_naming(args.hf_checkpoint)
    return AttachContext(
        lora=LoRAConfig.from_args(args, select_all=arch_spec.complete_layout),
        transformer_config=transformer_config,
        tp_size=ps.get_tensor_model_parallel_world_size(),
        tp_rank=ps.get_tensor_model_parallel_rank(),
        layer_prefix=layer_prefix,
        shared_expert=shared_expert,
    )


def _attach_layer(layer: nn.Module, arch_spec: LoRAArchSpec, context: AttachContext) -> int:
    hf_layer = f"{context.layer_prefix}{layer.layer_number - 1}."
    wrapped = 0
    attention = getattr(layer, "self_attention", None)
    if attention is not None:
        wrapped += _attach_attention(attention, hf_layer, arch_spec, context)

    mlp = layer.mlp
    if hasattr(mlp, "linear_fc1"):
        assert getattr(mlp.config, "gated_linear_unit", True), "native LoRA assumes a gated (SwiGLU) MLP"
        wrapped += arch_spec.mlp.attach(mlp, f"{hf_layer}mlp.", context)
    shared = getattr(mlp, "shared_experts", None)
    if shared is not None and hasattr(shared, "linear_fc1"):
        wrapped += arch_spec.mlp.attach(shared, f"{hf_layer}{context.shared_expert}", context)
    if arch_spec.experts is not None:
        wrapped += arch_spec.experts.attach(mlp, hf_layer, context)
    return wrapped


def _attach_attention(attention: nn.Module, hf_layer: str, arch_spec: LoRAArchSpec, context: AttachContext) -> int:
    hf_prefix = f"{hf_layer}{arch_spec.attention.hf_block}."
    attached = arch_spec.attention.attach(attention, hf_prefix, context)
    selected = sorted(name for name in arch_spec.attention.supported_targets if context.selects(hf_prefix + name))
    # Hybrid GDN mixer layers have no fused qkv; any other miss means the registry disagrees with the model.
    assert attached or not selected or arch_spec.allows_mixer_only_adapter_chunks, (
        f"native LoRA's {arch_spec.name} spec attached none of {selected} at {hf_prefix!r}; "
        "the built attention module does not match the registered architecture."
    )
    return attached


def _assert_supported_run(args) -> None:
    """Reject runtime interactions that have not been validated for native LoRA."""
    assert not args.overlap_param_gather, (
        "native LoRA does not support --overlap-param-gather: sibling-module adapter attachment has not "
        "been validated against MCore's bucket prefetch ordering."
    )
    assert not args.moe_shared_expert_overlap, (
        "native LoRA does not support --moe-shared-expert-overlap: the dispatcher owns the shared-expert "
        "communication, so the adapter's gather/reduce no longer matches the module's parallel mode."
    )
    assert not args.overlap_grad_reduce, (
        "native LoRA does not support --overlap-grad-reduce: replicated adapter gradients need a "
        "tensor-parallel sum over the buffer MCore's in-flight data-parallel reduce-scatter writes."
    )


def _require_grad_on_first_activation(model) -> nn.Module | None:
    """Make a frozen embedding output require grad so recomputation enters each block."""
    embedding = getattr(model, "embedding", None)
    if embedding is None:
        return None

    def hook(_module, _inputs, output):
        return output if output.requires_grad else output.requires_grad_(True)

    embedding.register_forward_hook(hook)
    return embedding


__all__ = [
    "apply_native_lora",
    "export_lora_hf_named",
    "export_lora_sglang_named",
    "load_lora_adapter_hf",
    "wrap_model_provider_with_lora",
]
