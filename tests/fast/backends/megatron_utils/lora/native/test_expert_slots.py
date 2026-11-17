"""Slot isolation on expert-major tokens, including empty slots and padded ranks."""

from contextvars import ContextVar
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


@pytest.mark.parametrize("projection", ["gate_up", "up", "down"])
@pytest.mark.parametrize("counts", [(3, 0, 2), (0, 0, 0)])
@pytest.mark.parametrize("shared_outer", [True, False])
def test_expert_major_forward_and_gradients_match_per_token_reference(monkeypatch, projection, counts, shared_outer):
    context = _context()
    context = replace(context, lora=replace(context.lora, experts_shared_outer_loras=shared_outer))
    common = dict(
        hf_prefix="model.layers.0.mlp.experts.",
        reference=torch.empty(()),
        context=context,
        num_local_experts=2,
        moe_intermediate=5,
        width=6,
        is_ep=False,
    )
    build = (
        (lambda: LoRAGroupedFC2(hf_leaf="down_proj", **common))
        if projection == "down"
        else (
            lambda: LoRAGroupedFC1(
                hf_leaves=("up_proj",) if projection == "up" else ("gate_proj", "up_proj"), **common
            )
        )
    )
    adapter = NativeMultiLoRAExpertAdapter(
        [build() for _ in counts], context, ContextVar("test_routing", default=None)
    )
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
    plan = expert_slots.build_expert_slot_routing(dispatcher, per_expert, counts, context, "cpu")
    x = torch.randn(len(permutation), 5 if projection == "down" else 6, requires_grad=True)
    token = adapter.routing_plan.set(plan)
    try:
        actual = adapter(x, nn.Identity())
    finally:
        adapter.routing_plan.reset(token)
    expected = []
    for row, slot, expert in zip(x, dispatched_slots.tolist(), [0] * sum(counts) + [1] * sum(counts), strict=True):
        single = adapter.adapters[slot]
        if projection == "down":
            b = single.w2_B if shared_outer else single.w2_B[expert]
            delta = b @ (single.w2_A[expert] @ row)
        else:
            delta = torch.cat([b[expert] @ ((a if shared_outer else a[expert]) @ row) for _, a, b in single.factors()])
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


def test_routing_plan_restores_changed_width_and_gradients():
    plan = expert_slots.ExpertSlotRouting(torch.tensor([2, 0, 3, 1]), ((1, 1), (1, 1)))
    x = torch.randn(4, 3, requires_grad=True)
    first, second = plan.group_by_slot(x)
    restored = plan.restore_order(torch.cat([2 * first[:, :2], 3 * second[:, :2]]))
    scales = torch.tensor([2.0, 3.0, 2.0, 3.0])[:, None]
    torch.testing.assert_close(restored, x[:, :2] * scales)
    restored.sum().backward()
    torch.testing.assert_close(x.grad, torch.cat([scales.expand(4, 2), torch.zeros(4, 1)], dim=1))


@pytest.mark.parametrize("exit_mode", ["normal", "exception", "nested"])
def test_expert_forward_scopes_one_plan_and_restores_the_previous_binding(monkeypatch, exit_mode):
    from megatron.core.transformer.moe import token_dispatcher

    class Dispatcher:
        tp_size = 1
        hidden_shape_before_permute = (2, 6)

    monkeypatch.setattr(token_dispatcher, "MoEAlltoAllTokenDispatcher", Dispatcher)
    monkeypatch.setattr(expert_slots, "_co_permute_slot_ids", lambda _d, ids, _n: ids)
    routing_plan = ContextVar("test_layer_routing", default=None)
    previous = expert_slots.ExpertSlotRouting(torch.arange(2), ((2,),))
    binding = routing_plan.set(previous)
    seen = []

    class Experts(nn.Module):
        def forward(self, hidden, per_expert):
            plan = routing_plan.get()
            assert plan is not None and plan is not previous
            seen.append(plan)
            if exit_mode == "exception":
                raise ValueError("expert failed")
            if exit_mode == "nested" and len(seen) == 1:
                hidden = self(hidden, per_expert)
                assert routing_plan.get() is plan
            return hidden * 2

    experts = Experts()
    mlp = SimpleNamespace(
        experts=experts,
        token_dispatcher=Dispatcher(),
        config=SimpleNamespace(moe_permute_fusion=False, moe_expert_capacity_factor=None),
    )
    reference = SimpleNamespace(token_counts_by_slot=(1, 1, 0), context=_context(), routing_plan=routing_plan)
    expert_slots.install_expert_slot_routing(mlp, reference)
    assert not experts._forward_pre_hooks
    try:
        if exit_mode == "exception":
            with pytest.raises(ValueError, match="expert failed"):
                experts(torch.randn(2, 6), [2])
        else:
            experts(torch.randn(2, 6), [2])
        assert routing_plan.get() is previous
        assert len(seen) == (2 if exit_mode == "nested" else 1)
        if exit_mode == "nested":
            assert seen[0] is not seen[1]
    finally:
        routing_plan.reset(binding)
