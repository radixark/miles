"""Single public entry point and lifecycle orchestration for native LoRA."""

from __future__ import annotations

import logging

import torch.nn as nn

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named, load_lora_adapter_hf
from miles.backends.megatron_utils.lora.native.naming import ConverterNaming
from miles.backends.megatron_utils.lora.native.registry import resolve_checkpoint_spec
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext, LoRAArchSpec
from miles.backends.megatron_utils.megatron_to_hf import converter_model_name
from miles.utils.hf_utils.config import load_hf_config
from miles.utils.hf_utils.weight_mapping import build_hf_namespace_predicate

logger = logging.getLogger(__name__)


def apply_native_lora(model, args):
    """Attach native LoRA to one model chunk before Float16Module/DDP wrapping."""
    model_type, arch_spec = resolve_checkpoint_spec(args.hf_checkpoint)
    context = _resolve_attach_context(args, model.config, arch_spec)
    arch_spec.validate(context)

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
        context.lora.rank,
        context.lora.alpha,
        context.lora.dropout,
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

    tp_size = ps.get_tensor_model_parallel_world_size()
    naming = arch_spec.declared_naming or ConverterNaming(
        args,
        converter_model_name(args, load_hf_config(args.hf_checkpoint)),
        tp_size,
        build_hf_namespace_predicate(args.hf_checkpoint),
    )
    return AttachContext(
        lora=LoRAConfig.from_args(
            args, select_all=arch_spec.fixed_targets is not None and arch_spec.fixed_targets.select_all
        ),
        transformer_config=transformer_config,
        tp_size=tp_size,
        tp_rank=ps.get_tensor_model_parallel_rank(),
        naming=naming,
    )


def _attach_layer(layer: nn.Module, arch_spec: LoRAArchSpec, context: AttachContext) -> int:
    attention = layer.self_attention
    # Hybrid GDN mixer layers have no fused qkv; any other miss means the registry disagrees with the model.
    assert arch_spec.has_linear_attention_layers or arch_spec.attention.applies_to(attention), (
        f"native LoRA's {arch_spec.name} spec does not apply to layer {layer.layer_number - 1}'s "
        f"{type(attention).__name__}; the built attention module does not match the registered architecture."
    )
    wrapped = arch_spec.attention.attach(attention, layer, context)

    # the MLP layout applies to a dense MLP or an MLP-shaped shared expert, not to the MoE block itself
    mlp = layer.mlp
    wrapped += arch_spec.mlp.attach(mlp, layer, context)
    shared = getattr(mlp, "shared_experts", None)  # absent on dense layers and MoEs without shared experts
    if shared is not None:
        wrapped += arch_spec.mlp.attach(shared, layer, context)
    if arch_spec.experts is not None:
        wrapped += arch_spec.experts.attach(mlp, layer, context)
    return wrapped


def _require_grad_on_first_activation(model) -> nn.Module | None:
    """Make a frozen embedding output require grad in training so recomputation enters each block."""
    embedding = getattr(model, "embedding", None)
    if embedding is None:
        return None

    def hook(module, _inputs, output):
        return output.requires_grad_(True) if module.training and not output.requires_grad else output

    embedding.register_forward_hook(hook)
    return embedding


__all__ = [
    "apply_native_lora",
    "export_lora_hf_named",
    "load_lora_adapter_hf",
    "wrap_model_provider_with_lora",
]
