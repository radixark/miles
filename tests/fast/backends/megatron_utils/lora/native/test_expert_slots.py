"""Slot isolation on expert-major tokens, including empty slots and padded ranks."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.modules import expert_slots
from miles.backends.megatron_utils.lora.native.modules.expert_slots import NativeMultiLoRAExpertAdapter
from miles.backends.megatron_utils.lora.native.modules.moe import LoRAGroupedFC1, LoRAGroupedFC2
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext


def _context():
    return AttachContext(
        lora=LoRAConfig(rank=4, alpha=8, dropout=0, target_modules=None, num_slots=3),
        transformer_config=SimpleNamespace(hidden_size=6, sequence_parallel=False),
        tp_size=1,
        tp_rank=0,
        naming=None,
    )


@pytest.mark.parametrize("projection", ["gate_up", "down"])
@pytest.mark.parametrize("counts", [(3, 0, 2), (0, 0, 0)])
def test_expert_major_forward_and_gradients_match_per_token_reference(monkeypatch, projection, counts):
    context = _context()
    common = dict(
        hf_prefix="model.layers.0.mlp.experts.",
        reference=torch.empty(()),
        context=context,
        num_local_experts=2,
        moe_intermediate=5,
        width=6,
        is_ep=False,
    )
    build = (lambda: LoRAGroupedFC2(**common)) if projection == "down" else (lambda: LoRAGroupedFC1(**common))
    adapter = NativeMultiLoRAExpertAdapter([build() for _ in counts], context)
    for slot in range(3):
        adapter.init_slot(slot, slot + 1, 6)
        with torch.no_grad():
            for name, parameter in adapter.adapters[slot].named_parameters():
                if name.endswith("_B"):
                    parameter.normal_()
                    parameter[..., slot + 1 :].zero_()
    slot_ids = torch.repeat_interleave(torch.arange(3), torch.tensor(counts))
    # top-2 dispatch duplicates each token once into each expert's segment.
    permutation = torch.arange(sum(counts)).repeat(2)
    dispatched_slots = slot_ids[permutation]
    per_expert = [sum(counts), sum(counts)]
    monkeypatch.setattr(expert_slots, "_co_permute_slot_ids", lambda _d, ids, _n: ids[permutation])
    dispatcher = SimpleNamespace(hidden_shape_before_permute=(sum(counts), 6))
    adapter.routing = expert_slots.build_expert_slot_routing(dispatcher, per_expert, counts, context, "cpu")
    x = torch.randn(len(permutation), 5 if projection == "down" else 6, requires_grad=True)
    actual = adapter(x, nn.Identity())
    expected = []
    for row, slot, expert in zip(x, dispatched_slots.tolist(), [0] * sum(counts) + [1] * sum(counts), strict=True):
        single = adapter.adapters[slot]
        if projection == "down":
            b = single.w2_B
            delta = b @ (single.w2_A[expert] @ row)
        else:
            delta = torch.cat([single.w1_B[expert] @ (single.w1_A @ row), single.w3_B[expert] @ (single.w3_A @ row)])
        expected.append((6 / (slot + 1)) * delta)
    reference = torch.stack(expected) if expected else actual.new_empty(actual.shape)
    torch.testing.assert_close(actual, reference)
    actual.sum().backward()
    for single in adapter.adapters:
        for name, parameter in single.named_parameters():
            assert parameter.grad is not None
            rank = adapter.lora_ranks_by_slot[list(adapter.adapters).index(single)]
            padded = parameter[..., rank:, :] if name.endswith("_A") else parameter[..., rank:]
            assert torch.count_nonzero(padded) == 0


def test_expert_routing_narrows_the_sequence_parallel_shard(monkeypatch):
    context = _context()
    context = replace(
        context, tp_size=2, tp_rank=1, transformer_config=SimpleNamespace(hidden_size=6, sequence_parallel=True)
    )
    seen = []

    def co_permute(_dispatcher, ids, _experts):
        seen.extend(ids.tolist())
        return ids

    monkeypatch.setattr(expert_slots, "_co_permute_slot_ids", co_permute)
    routing = expert_slots.build_expert_slot_routing(
        SimpleNamespace(hidden_shape_before_permute=(3, 6)), [3], (2, 4, 0), context, "cpu"
    )
    assert seen == [1, 1, 1]
    assert routing.counts == ((0,), (3,), (0,))
