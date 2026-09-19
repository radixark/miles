"""MoE per-layer policy specs: what happens when MLP targets meet an expert layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from miles_plugins.lora.modules.linear import attach_adapter_forward
from miles_plugins.lora.modules.moe import LoRAGroupedFC1, LoRAGroupedFC2, LoRASharedExpertsAdapter
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


class InklingMoESpec:
    """Routed grouped experts (shared-A / per-expert-B) plus shared sub-experts."""

    supported_targets: frozenset[str] = frozenset()

    def validate_layer(self, mlp: nn.Module, context: AttachContext) -> frozenset[str]:
        del mlp, context
        return _NO_TARGETS

    def attach(self, mlp: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        from megatron.core import parallel_state

        if not hasattr(mlp, "experts"):
            return 0
        config = mlp.config
        assert (getattr(config, "expert_tensor_parallel_size", 1) or 1) == 1, "Inkling LoRA assumes ETP=1"
        experts = mlp.experts
        is_ep = parallel_state.get_expert_model_parallel_world_size() > 1
        count = 0

        fc1_adapter = LoRAGroupedFC1(
            hf_prefix=hf_prefix + "mlp.experts.",
            reference=experts.linear_fc1.weight0,
            context=context,
            num_local_experts=experts.num_local_experts,
            moe_intermediate=config.moe_ffn_hidden_size,
            is_ep=is_ep,
        )
        experts.lora_fc1_adapter = fc1_adapter
        attach_adapter_forward(experts.linear_fc1, fc1_adapter, context.scale)
        count += 1

        fc2_adapter = LoRAGroupedFC2(
            hf_prefix=hf_prefix + "mlp.experts.",
            reference=experts.linear_fc2.weight0,
            context=context,
            num_local_experts=experts.num_local_experts,
            moe_intermediate=config.moe_ffn_hidden_size,
            is_ep=is_ep,
        )
        experts.lora_fc2_adapter = fc2_adapter
        attach_adapter_forward(experts.linear_fc2, fc2_adapter, context.scale)
        count += 1

        shared = getattr(mlp, "shared_experts", None)
        if shared is not None:
            count += self._attach_shared(shared, hf_prefix, context)
        return count

    @staticmethod
    def _attach_shared(shared: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        subs = list(shared.experts)
        local_intermediate = shared.experts[0].linear_fc1.weight.shape[0] // 2
        adapter = LoRASharedExpertsAdapter(
            hf_prefix=hf_prefix + "mlp.shared_experts.",
            fc1_reference=subs[0].linear_fc1.weight,
            fc2_reference=subs[0].linear_fc2.weight,
            context=context,
            num_shared=len(subs),
            local_intermediate=local_intermediate,
        )
        shared.lora_adapter = adapter

        for index, sub in enumerate(subs):
            for host_attr, delta in (("linear_fc1", adapter.fc1_delta), ("linear_fc2", adapter.fc2_delta)):
                host = getattr(sub, host_attr)
                original = host.forward

                def forward(x, *args, _original=original, _host=host, _delta=delta, _index=index, **kwargs):
                    out, bias = _original(x, *args, **kwargs)
                    return torch.add(out, _delta(x, _host, _index), alpha=context.scale), bias

                host.forward = forward
        return 1
