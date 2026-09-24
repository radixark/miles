"""Routed-expert native-LoRA specs."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn

from miles_plugins.lora.modules.linear import attach_adapter_forward, attach_delta_forward
from miles_plugins.lora.modules.moe import LoRAGroupedFC1, LoRAGroupedFC2, LoRASharedExpertsAdapter
from miles_plugins.lora.spec.base import AttachContext


class InklingExpertsSpec:
    """Routed grouped experts (shared-A / per-expert-B) plus shared sub-experts."""

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
                attach_delta_forward(host, functools.partial(_indexed, delta, index), context.scale)
                adapter.bind_host(host)
        return 1


def _indexed(delta, index: int, x: torch.Tensor, host: nn.Module, *_host_args) -> torch.Tensor:
    return delta(x, host, index)
