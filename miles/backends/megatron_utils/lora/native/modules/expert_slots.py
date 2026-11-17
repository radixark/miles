# SPDX-License-Identifier: Apache-2.0
"""Native multi-LoRA routing for Megatron's CUDA all-to-all expert dispatcher.

Adapter-slot co-permutation adapted from Megatron-Bridge; see NOTICE.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.native.modules.linear import NativeMultiLoRASlots


@dataclass(frozen=True)
class ExpertSlotRouting:
    permutation: torch.Tensor
    counts: tuple[tuple[int, ...], ...]

    def group_by_slot(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Group dispatched rows by slot, preserving expert order within each slot."""
        rows = x.index_select(0, self.permutation)
        return rows.split([sum(per_expert) for per_expert in self.counts])

    def restore_order(self, deltas: torch.Tensor) -> torch.Tensor:
        """Scatter grouped deltas back to the dispatcher's row order."""
        return torch.empty_like(deltas).index_copy_(0, self.permutation, deltas)


def _co_permute_slot_ids(dispatcher, slot_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    # Megatron is optional on CPU hosts that import the native adapter definitions.
    from megatron.core.tensor_parallel.mappings import all_to_all
    from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs

    slot_ids = slot_ids.index_select(0, dispatcher.reversed_local_input_permutation_mapping)
    if dispatcher.ep_size > 1:
        slot_ids = all_to_all(
            dispatcher.ep_group, slot_ids, list(dispatcher.output_splits), list(dispatcher.input_splits)
        )
    if num_experts > 1:
        slot_ids, _ = sort_chunks_by_idxs(
            slot_ids,
            dispatcher.num_global_tokens_per_local_expert.ravel(),
            dispatcher.sort_input_by_local_experts,
            fused=False,
        )
    return slot_ids


def build_expert_slot_routing(dispatcher, per_expert, counts, context, device) -> ExpertSlotRouting:
    slot_ids = torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), torch.tensor(counts, device=device)
    )
    local_tokens = dispatcher.hidden_shape_before_permute[0]
    if len(slot_ids) != local_tokens:
        assert (
            context.sequence_parallel and len(slot_ids) == local_tokens * context.tp_size
        ), f"{len(slot_ids)} routed tokens cannot cover {local_tokens} local MoE tokens"
        slot_ids = slot_ids.narrow(0, context.tp_rank * local_tokens, local_tokens)
    num_experts = len(per_expert)
    slot_ids = _co_permute_slot_ids(dispatcher, slot_ids, num_experts)
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=device), torch.as_tensor(per_expert, device=device)
    )
    assert len(expert_ids) == len(slot_ids), "dispatched token and adapter-slot counts disagree"
    keys = slot_ids.long() * num_experts + expert_ids
    permutation = torch.argsort(keys, stable=True)
    slot_counts = torch.bincount(keys, minlength=len(counts) * num_experts).view(len(counts), num_experts)
    return ExpertSlotRouting(permutation, tuple(tuple(row) for row in slot_counts.tolist()))


class NativeMultiLoRAExpertAdapter(NativeMultiLoRASlots):
    """Expert slots sharing the MoE layer's scoped routing plan."""

    def __init__(self, adapters, context, routing_plan: ContextVar[ExpertSlotRouting | None]):
        assert all(type(adapter) is type(adapters[0]) for adapter in adapters)
        super().__init__(adapters, context)
        self.routing_plan = routing_plan

    def forward(self, x: torch.Tensor, base_module: nn.Module, *_host_args) -> torch.Tensor:
        routing = self.routing_plan.get()
        assert routing is not None, "expert adapters must run inside the routed expert forward"
        segments = routing.group_by_slot(x)
        assert len(segments) == len(self.adapters), "the routing plan must cover every adapter slot"
        deltas = []
        for slot, adapter in enumerate(self.adapters):
            delta = adapter(segments[slot], base_module, routing.counts[slot])
            scale = self.lora_alphas_by_slot[slot] / self.lora_ranks_by_slot[slot]
            deltas.append(delta * scale)
        return routing.restore_order(torch.cat(deltas))

    def add_delta(self, out: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return out + delta


def install_expert_slot_routing(mlp: nn.Module, reference: NativeMultiLoRAExpertAdapter) -> None:
    """Bind one plan around the expert forward, shared by FC1/FC2 and reset even on recomputation exits."""
    # Dispatcher internals and kernels are available only in the Megatron runtime.
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

    dispatcher = mlp.token_dispatcher
    assert isinstance(dispatcher, MoEAlltoAllTokenDispatcher), "native multi-LoRA experts require alltoall dispatch"
    assert not mlp.config.moe_permute_fusion, "native multi-LoRA experts require moe_permute_fusion=False"
    assert dispatcher.tp_size == 1, "native multi-LoRA experts require expert tensor parallel size 1"
    assert not mlp.config.moe_expert_capacity_factor, "native multi-LoRA experts do not support capacity padding"

    expert_forward = mlp.experts.forward

    @wraps(expert_forward)
    def forward_with_routing(hidden, per_expert, *args, **kwargs):
        counts = reference.token_counts_by_slot
        assert counts is not None, "call AdapterSlots.route_tokens before the MoE forward"
        plan = build_expert_slot_routing(dispatcher, per_expert, counts, reference.context, hidden.device)
        token = reference.routing_plan.set(plan)
        try:
            return expert_forward(hidden, per_expert, *args, **kwargs)
        finally:
            reference.routing_plan.reset(token)

    mlp.experts.forward = forward_with_routing
