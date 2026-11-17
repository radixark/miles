# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native multi-LoRA routing for Megatron's CUDA all-to-all expert dispatcher.

The co-permutation follows Megatron-Bridge's Apache-2.0 multi_lora_layers.py.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.native.modules.linear import NativeMultiLoRAAdapter


@dataclass(frozen=True)
class ExpertSlotRouting:
    permutation: torch.Tensor
    inverse: torch.Tensor
    counts: tuple[tuple[int, ...], ...]


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
    inverse = torch.empty_like(permutation)
    inverse.scatter_(0, permutation, torch.arange(len(permutation), device=device))
    slot_counts = torch.bincount(keys, minlength=len(counts) * num_experts).view(len(counts), num_experts)
    return ExpertSlotRouting(permutation, inverse, tuple(tuple(row) for row in slot_counts.tolist()))


class NativeMultiLoRAExpertAdapter(NativeMultiLoRAAdapter):
    """Reuse native slot lifecycle and IO, with tokens grouped by both slot and expert."""

    routing: ExpertSlotRouting | None = None

    def _validate_adapter_types(self, adapters) -> None:
        assert all(type(adapter) is type(adapters[0]) for adapter in adapters)

    def forward(self, x: torch.Tensor, base_module: nn.Module, *_host_args) -> torch.Tensor:
        routing = self.routing
        assert routing is not None, "routed expert adapters need the MoE slot-routing hook"
        rows = x.index_select(0, routing.permutation)
        segments = rows.split([sum(counts) for counts in routing.counts])
        deltas = [
            adapter(segment, base_module, counts) * (alpha / rank)
            for adapter, segment, counts, alpha, rank in zip(
                self.adapters, segments, routing.counts, self.lora_alphas_by_slot, self.lora_ranks_by_slot, strict=True
            )
        ]
        return torch.cat(deltas).index_select(0, routing.inverse)

    def add_delta(self, out: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return out + delta


def install_expert_slot_routing(mlp: nn.Module, adapters: list[NativeMultiLoRAExpertAdapter]) -> None:
    # Dispatcher internals and kernels are available only in the Megatron runtime.
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

    dispatcher = mlp.token_dispatcher
    assert isinstance(dispatcher, MoEAlltoAllTokenDispatcher), "native multi-LoRA experts require alltoall dispatch"
    assert not mlp.config.moe_permute_fusion, "native multi-LoRA experts require moe_permute_fusion=False"
    assert dispatcher.tp_size == 1, "native multi-LoRA experts require expert tensor parallel size 1"
    assert not mlp.config.moe_expert_capacity_factor, "native multi-LoRA experts do not support capacity padding"

    def route(_experts, inputs):
        hidden, per_expert = inputs[:2]
        reference = adapters[0]
        counts = reference.token_counts_by_slot
        assert counts is not None, "call AdapterSlots.route_tokens before the MoE forward"
        routing = build_expert_slot_routing(dispatcher, per_expert, counts, reference.context, hidden.device)
        for adapter in adapters:
            adapter.routing = routing

    mlp.experts.register_forward_pre_hook(route)
