"""Slot-capacity resolution: the flag-derived byte model, the residency accounting,
the probe sequence, and the min-of-ranks arithmetic; the GPU probe itself is
exercised on hardware."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.backends.megatron_utils.lora import slot_capacity
from miles.backends.megatron_utils.lora.slot_capacity import (
    PROBE_SLOT,
    RankProbe,
    bytes_per_train_param,
    expert_data_parallel_size,
    predicted_slot_bytes,
    probe_slot_capacity,
    resident_slot_bytes,
    resolve_slot_capacity,
)

GIB = 1 << 30


@pytest.mark.parametrize(
    ("bf16", "fp16", "accum_fp32", "expected"),
    [
        (True, False, True, 18),  # bf16 weight + fp32 grad + fp32 master + moments
        (True, False, False, 16),  # grad follows the bf16 weight
        (False, False, True, 16),  # pure fp32: no separate master
        (False, True, True, 18),
    ],
    ids=["bf16-accum", "bf16-raw-grad", "fp32", "fp16-accum"],
)
def test_bytes_per_train_param_follows_the_precision_flags(bf16, fp16, accum_fp32, expected):
    args = Namespace(bf16=bf16, fp16=fp16, accumulate_allreduce_grads_in_fp32=accum_fp32)
    assert bytes_per_train_param(args) == expected


def test_optimizer_state_is_split_across_data_parallel_ranks():
    # bf16 weight + fp32 grad on every rank; the fp32 master and moments (12 B) are scattered over DP=2
    args = Namespace(bf16=True, fp16=False, accumulate_allreduce_grads_in_fp32=True)
    assert bytes_per_train_param(args, dp_size=2) == 12


def test_expert_adapters_share_optimizer_state_over_the_expert_data_parallel_ranks():
    # 8 GPUs, TP2 / EP8 / ETP1: DP=4 for dense params, but every rank owns its experts alone
    args = Namespace(
        bf16=True,
        fp16=False,
        accumulate_allreduce_grads_in_fp32=True,
        tensor_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
    )
    assert expert_data_parallel_size(args, dp_size=4) == 1
    probe = RankProbe(free=0, slot_bytes=0, act_peak=0, adapter_local_params=100_000, adapter_expert_params=99_000)
    # dense: 1_000 * (6 + 12 / 4); expert: 99_000 * (6 + 12 / 1)
    assert predicted_slot_bytes(args, probe, dp_size=4) == 1_000 * 9 + 99_000 * 18
    # twice the GPUs (DP=8) halve both shares' state, the experts' now over two ranks
    assert expert_data_parallel_size(args, dp_size=8) == 2
    assert predicted_slot_bytes(args, probe, dp_size=8) == 1_000 * 7.5 + 99_000 * 12


def _probe(free=100 * GIB, slot=2 * GIB, act_peak=10 * GIB) -> RankProbe:
    return RankProbe(free=free, slot_bytes=slot, act_peak=act_peak, adapter_local_params=slot // 18)


def _args(**overrides) -> Namespace:
    defaults = dict(lora_rank=32, train_memory_margin_bytes=2 * GIB)
    return Namespace(**{**defaults, **overrides})


def test_the_worst_rank_bounds_the_capacity():
    probes = [_probe(), _probe(free=48 * GIB)]
    # worst rank: the probe slot's own 2 GiB is head-room too: (48 + 2 - 10 - 2) / 2 = 19 slots
    assert resolve_slot_capacity(_args(), probes) == 19


def test_the_grouped_gemm_limit_bounds_expert_slots():
    # memory would allow (200 + 2 - 10 - 2) / 2 = 95 slots; 16 local experts per slot cap it at 1023 // 16
    roomy = RankProbe(
        free=200 * GIB, slot_bytes=2 * GIB, act_peak=10 * GIB, adapter_local_params=1, expert_groups_per_slot=16
    )
    assert resolve_slot_capacity(_args(), [roomy]) == 63
    # a tighter rank keeps memory the binding constraint
    tight = RankProbe(
        free=40 * GIB, slot_bytes=2 * GIB, act_peak=10 * GIB, adapter_local_params=1, expert_groups_per_slot=16
    )
    assert resolve_slot_capacity(_args(), [roomy, tight]) == 15
    # dense adapters (no expert groups) are memory-bound only
    assert resolve_slot_capacity(_args(), [_probe()]) == 45


def _engine_args(**overrides) -> Namespace:
    defaults = dict(
        lora_rank=16,
        train_memory_margin_bytes=GIB,
        multi_lora_rollout_seqs_per_slot=16,
        multi_lora_rollout_tokens_per_seq=8192,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=2,
        sglang_ep_size=2,
        sglang_mem_fraction_static=0.9,
        num_layers=48,
        group_query_attention=True,
        num_query_groups=4,
        num_attention_heads=32,
        kv_channels=128,
        hidden_size=2048,
        seq_length=8192,
    )
    return Namespace(**{**defaults, **overrides})


def test_the_engine_bound_fits_adapter_buffers_and_kv_for_every_slot():
    from miles.backends.megatron_utils.lora.slot_capacity import engine_slot_capacity

    probe = RankProbe(
        free=200 * GIB,
        slot_bytes=GIB,
        act_peak=GIB,
        adapter_local_params=1,
        gpu_total_bytes=140 * GIB,
        base_dense_params=2_000_000_000,  # 2B dense, 28B expert params: a 30B MoE
        base_expert_params=28_000_000_000,
        adapter_dense_params=10_000_000,
        adapter_expert_params_total=630_000_000,
    )
    n, detail = engine_slot_capacity(_engine_args(), probe)
    # weights per engine GPU: 2 B * (2B / TP2 + 28B / EP2) = 30 GB; budget = 0.9 * 140 GiB - 30 GB
    # adapter per GPU: 2 B * (10M / 2 + 630M / 2) = 640 MB; KV per token per GPU: 48 * 2 * 128 * 2 * 2 = 48 KiB
    # KV per slot: 16 seqs * 8192 tokens * 48 KiB / 4 engines = 1.5 GiB
    budget = 0.9 * 140 * GIB - 2 * (2e9 / 2 + 28e9 / 2)
    per_slot = 2 * (10e6 / 2 + 630e6 / 2) + 16 * 8192 * 49152 / 4
    assert n == int(budget // per_slot)
    assert detail["engines"] == 4 and detail["seqs_per_slot"] == 16
    # switched off by default
    assert engine_slot_capacity(_engine_args(multi_lora_rollout_seqs_per_slot=0), probe) is None
    # and it is the binding constraint when smaller than memory and the group limit
    roomy = RankProbe(**{**probe.__dict__, "expert_groups_per_slot": 8})
    assert resolve_slot_capacity(_engine_args(), [roomy]) == min(n, 1023 // 8)


def test_no_room_for_one_slot_is_a_launch_error():
    probes = [_probe(free=9 * GIB, slot=2 * GIB, act_peak=10 * GIB)]  # (9 + 2 - 10 - 2) / 2 < 1
    with pytest.raises(AssertionError, match="Lower --lora-rank"):
        resolve_slot_capacity(_args(), probes)


def test_capacity_never_goes_negative():
    assert _probe(free=3 * GIB, act_peak=10 * GIB).capacity(margin_bytes=0) == 0


@pytest.mark.parametrize("slot", [0, -GIB])
def test_a_non_positive_residency_is_not_a_capacity(slot):
    with pytest.raises(ValueError, match="non-positive slot residency"):
        _probe(slot=slot).capacity(margin_bytes=0)


def _cuda_tensor(pointer: int, nbytes: int) -> SimpleNamespace:
    return SimpleNamespace(
        is_cuda=True,
        device=SimpleNamespace(index=0),
        grad=None,
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: pointer, nbytes=lambda: nbytes),
    )


def test_residency_counts_each_storage_once(monkeypatch):
    weight_a, weight_b = _cuda_tensor(1, 20), _cuda_tensor(2, 30)
    # both main_grads are views into one padded grad buffer: count that buffer once
    weight_a.main_grad = weight_b.main_grad = _cuda_tensor(3, 128)
    master = _cuda_tensor(4, 100)
    weight_a.main_param = master
    child = SimpleNamespace(
        get_parameters=lambda: [master],  # the same master the param points at
        optimizer=SimpleNamespace(
            state={0: {"exp_avg": _cuda_tensor(5, 100), "exp_avg_sq": _cuda_tensor(6, 100), "step": 1}}
        ),
    )
    monkeypatch.setattr(slot_capacity, "adapter_slot_parameters", lambda model, slot: [weight_a, weight_b])
    monkeypatch.setattr(slot_capacity, "_slot_children", lambda optimizer, slot: [child])

    assert resident_slot_bytes(model=None, optimizer=None, slot=PROBE_SLOT) == 20 + 30 + 128 + 100 + 100 + 100


def _backend(calls, step_outcome) -> SimpleNamespace:
    return SimpleNamespace(
        load_slot=AsyncMock(side_effect=lambda *a: calls.append("load")),
        unload_slot=AsyncMock(side_effect=lambda *a: calls.append("unload")),
        forward_backward=AsyncMock(side_effect=lambda *a: calls.append("fb")),
        optim_step=AsyncMock(side_effect=lambda *a: calls.append("step") or {PROBE_SLOT: step_outcome}),
    )


async def test_the_probe_warms_up_then_measures_one_max_size_step():
    calls = []
    snapshot = {
        "free": 50 * GIB,
        "slot_bytes": 2 * GIB,
        "act_peak": 10 * GIB,
        "adapter_local_params": 2 * GIB // 12,
        "adapter_expert_params": 0,
    }
    trainer = SimpleNamespace(
        multi_lora_memory_probe=AsyncMock(
            side_effect=lambda phase: calls.append(phase) or ([{}] if phase == "reset" else [snapshot])
        )
    )
    backend = _backend(calls, {"grad_norm": 0.5})
    args = _args(
        bf16=True, fp16=False, accumulate_allreduce_grads_in_fp32=True, lora_alpha=None, max_tokens_per_gpu=8192
    )

    probes = await probe_slot_capacity(args, backend, trainer, dp_size=2)

    assert calls == ["load", "fb", "step", "reset", "fb", "step", "measure", "unload"]
    assert probes == [RankProbe(**snapshot)]
    backend.load_slot.assert_awaited_once_with(PROBE_SLOT, 32, 64.0)  # alpha defaults to 2 * rank
    for call in backend.forward_backward.await_args_list:
        ((slot, row),) = call.args[1]
        assert slot == PROBE_SLOT
        assert len(row["tokens"]) == 8192  # exactly the budget one GPU admits per micro-batch
        assert row["target_len"] == len(row["weights"]) == 8191


async def test_an_unsettled_probe_step_is_not_a_measurement():
    backend = _backend([], {"skipped_nonfinite": True})
    trainer = SimpleNamespace(multi_lora_memory_probe=AsyncMock(return_value=[{}]))
    with pytest.raises(AssertionError, match="did not settle"):
        await probe_slot_capacity(_args(lora_alpha=64, max_tokens_per_gpu=16), backend, trainer)
