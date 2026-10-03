"""Routed-expert native-LoRA specs."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn

from miles_plugins.lora.modules.linear import attach_adapter_forward, attach_delta_forward
from miles_plugins.lora.modules.moe import LoRAGroupedFC1, LoRAGroupedFC2, LoRASharedExpertsAdapter
from miles_plugins.lora.spec.base import AttachContext


class GroupedExpertsSpec:
    """Shared-outer routed experts (shared A / per-expert B for gate/up, the reverse for down).

    ``block`` is the HF name of the routed-expert block under a layer. The adapter widths
    come from the grouped host weights, so latent MoEs need no extra table entry. Shared
    experts stored as a list of sub-experts get one adapter over all of them here; a plain
    shared-expert MLP is adapted by the architecture's MLP spec instead.
    """

    def __init__(self, block: str):
        self.block = block

    def attach(self, mlp: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        from megatron.core import parallel_state

        if not hasattr(mlp, "experts"):
            return 0
        config = mlp.config
        assert (getattr(config, "expert_tensor_parallel_size", 1) or 1) == 1, "native expert LoRA assumes ETP=1"
        experts = mlp.experts
        prefix = hf_prefix + self.block
        common = dict(
            hf_prefix=prefix,
            context=context,
            num_local_experts=experts.num_local_experts,
            moe_intermediate=config.moe_ffn_hidden_size,
            is_ep=parallel_state.get_expert_model_parallel_world_size() > 1,
        )
        count = 0
        if context.selects(f"{prefix}0.w1") or context.selects(f"{prefix}0.w3"):
            fc1 = experts.linear_fc1
            experts.lora_fc1_adapter = LoRAGroupedFC1(reference=fc1.weight0, width=fc1.weight0.shape[1], **common)
            attach_adapter_forward(fc1, experts.lora_fc1_adapter, context.scale)
            count += 1
        if context.selects(f"{prefix}0.w2"):
            fc2 = experts.linear_fc2
            experts.lora_fc2_adapter = LoRAGroupedFC2(reference=fc2.weight0, width=fc2.weight0.shape[0], **common)
            attach_adapter_forward(fc2, experts.lora_fc2_adapter, context.scale)
            count += 1

        shared = getattr(mlp, "shared_experts", None)
        if shared is not None and hasattr(shared, "experts"):
            count += self._attach_shared(shared, hf_prefix, context)
        return count

    @staticmethod
    def _attach_shared(shared: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        subs = list(shared.experts)
        local_intermediate = shared.experts[0].linear_fc1.weight.shape[0] // 2
        adapter = LoRASharedExpertsAdapter(
            hf_prefix=hf_prefix + context.shared_expert,
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
