"""MoE architecture specifications for Miles-native LoRA.

The MoE spec's job is per-layer policy: decide what happens when MLP targets
meet an expert layer. The MLP target names are injected at construction from
the arch spec's MLP spec, so there is a single source of truth for them.

Unsupported:

- EP-shared and shared-outer routed/grouped-expert LoRA.
- Sequential per-expert and router LoRA.
- Expert-TP/EP coordination and expert-axis HF/SGLang export.

TODO:

- Add MoE attachment support and expert-TP/EP context.
- Implement expert adapters, synchronization, HF export support, and SGLang
  packing.
"""

from __future__ import annotations

import torch.nn as nn

from miles_plugins.lora.spec.base import AttachContext

_NO_TARGETS: frozenset[str] = frozenset()


class GeneralExpertMoESpec:
    """MoE layers without a shared expert: routed/grouped experts only.

    Routed-expert LoRA is not natively supported, so MLP targets cannot attach
    anywhere on such a layer.
    """

    supported_targets: frozenset[str] = frozenset()

    def __init__(self, mlp_targets: frozenset[str]):
        self._mlp_targets = mlp_targets

    def validate_layer(self, mlp: nn.Module, context: AttachContext) -> frozenset[str]:
        """Return the MLP targets this layer cannot attach; raise when that is an error.

        Parser-added all-linear names mirror the MLA generic-qkv normalization:
        the layer skips what the architecture cannot attach and reports the
        skipped set for the orchestrator to log once per run.
        """
        if not hasattr(mlp, "experts") or not context.targets.intersection(self._mlp_targets):
            return _NO_TARGETS
        if context.lora.expanded_from_all_linear:
            return context.targets.intersection(self._mlp_targets)
        raise AssertionError(
            "Miles-native LoRA does not yet support routed/grouped expert projections, and this MoE "
            "layer has no attachable shared expert. Attention-only LoRA is supported for this model; "
            "for expert gate/up/down LoRA, use --megatron-to-hf-mode bridge or a model-specific "
            "--lora-provider-path."
        )


class SharedOuterExpertMoESpec(GeneralExpertMoESpec):
    """MoE layers with a shared (outer) expert: LoRA adapts the shared expert's MLP.

    Layers without a shared expert fall back to the routed-only policy, so one
    registry entry covers models that mix both layer kinds.
    """

    def validate_layer(self, mlp: nn.Module, context: AttachContext) -> frozenset[str]:
        if not hasattr(mlp, "experts") or not context.targets.intersection(self._mlp_targets):
            return _NO_TARGETS
        shared = getattr(mlp, "shared_experts", None)
        if shared is not None and hasattr(shared, "linear_fc1"):
            return _NO_TARGETS
        return super().validate_layer(mlp, context)
