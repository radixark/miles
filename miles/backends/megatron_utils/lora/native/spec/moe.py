"""Routed-expert native-LoRA specs."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.native.modules.linear import attach_adapter_forward, attach_delta_forward
from miles.backends.megatron_utils.lora.native.modules.moe import (
    LoRAGroupedFC1,
    LoRAGroupedFC2,
    LoRASharedExpertsAdapter,
)
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext


class GroupedExpertsSpec:
    """Shared-outer routed experts (shared A / per-expert B for gate/up, the reverse for down).

    The adapter widths come from the grouped host weights, so latent MoEs need no extra
    table entry. Shared experts stored as a list of sub-experts get one adapter over all of
    them here; a plain shared-expert MLP is adapted by the architecture's MLP spec instead.
    """

    def attach(self, mlp: nn.Module, layer: nn.Module, context: AttachContext) -> int:
        from megatron.core import parallel_state

        if not hasattr(
            mlp, "experts"
        ):  # config-access-exempt: distinguish routed-expert MLP hosts from dense MLP hosts
            return 0
        config = mlp.config
        experts = mlp.experts
        fc1, fc2 = experts.linear_fc1, experts.linear_fc2
        common = dict(
            context=context,
            num_local_experts=experts.num_local_experts,
            moe_intermediate=config.moe_ffn_hidden_size,
            is_ep=parallel_state.get_expert_model_parallel_world_size() > 1,
        )
        count = 0
        fc1_prefix = context.naming.block_prefix(layer, fc1.weight0, ["w1", "w3"])
        # selectors address one expert ("experts.*.w1"); shared-outer adapters cover them all
        if context.lora.selects(f"{fc1_prefix}0.w1") or context.lora.selects(f"{fc1_prefix}0.w3"):
            experts.lora_fc1_adapter = LoRAGroupedFC1(
                hf_prefix=fc1_prefix, reference=fc1.weight0, width=fc1.weight0.shape[1], **common
            )
            attach_adapter_forward(fc1, experts.lora_fc1_adapter, context.lora.scale)
            count += 1
        fc2_prefix = context.naming.block_prefix(layer, fc2.weight0, ["w2"])
        if context.lora.selects(f"{fc2_prefix}0.w2"):
            experts.lora_fc2_adapter = LoRAGroupedFC2(
                hf_prefix=fc2_prefix, reference=fc2.weight0, width=fc2.weight0.shape[0], **common
            )
            attach_adapter_forward(fc2, experts.lora_fc2_adapter, context.lora.scale)
            count += 1

        shared = getattr(
            mlp, "shared_experts", None
        )  # absent on MoEs without shared experts  # config-access-exempt: MoE implementations may omit shared experts
        if shared is not None and hasattr(
            shared, "experts"
        ):  # config-access-exempt: shared experts may themselves use a grouped expert wrapper
            count += self._attach_shared(shared, layer, context)
        return count

    @staticmethod
    def _attach_shared(shared: nn.Module, layer: nn.Module, context: AttachContext) -> int:
        subs = list(shared.experts)
        local_intermediate = shared.experts[0].linear_fc1.weight.shape[0] // 2
        adapter = LoRASharedExpertsAdapter(
            hf_prefix=context.naming.block_prefix(layer, subs[0].linear_fc1.weight, ["w1", "w3", "w2"]),
            fc1_reference=subs[0].linear_fc1.weight,
            fc2_reference=subs[0].linear_fc2.weight,
            context=context,
            num_shared=len(subs),
            local_intermediate=local_intermediate,
        )
        shared.lora_adapter = adapter

        for index, sub in enumerate(subs):
            for host_attr, delta in (("linear_fc1", adapter.fc1_delta), ("linear_fc2", adapter.fc2_delta)):
                host = getattr(
                    sub, host_attr
                )  # config-access-exempt: the expert architecture selects the host projection attribute
                attach_delta_forward(host, functools.partial(_sub_expert_delta, delta, index), context.lora.scale)
                adapter.bind_host(host)
        return 1


def _sub_expert_delta(delta, index: int, x: torch.Tensor, host: nn.Module, *_host_args) -> torch.Tensor:
    return delta(x, host, index)
