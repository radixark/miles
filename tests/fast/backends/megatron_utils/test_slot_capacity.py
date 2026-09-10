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


def _probe(free=100 * GIB, slot=2 * GIB, act_peak=10 * GIB) -> RankProbe:
    return RankProbe(free=free, slot_bytes=slot, act_peak=act_peak, adapter_local_params=slot // 18)


def _args(**overrides) -> Namespace:
    defaults = dict(lora_rank=32, train_memory_margin_bytes=2 * GIB)
    return Namespace(**{**defaults, **overrides})


def test_the_worst_rank_bounds_the_capacity():
    probes = [_probe(), _probe(free=48 * GIB)]
    # worst rank: the probe slot's own 2 GiB is head-room too: (48 + 2 - 10 - 2) / 2 = 19 slots
    assert resolve_slot_capacity(_args(), probes) == 19


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
    snapshot = {"free": 50 * GIB, "slot_bytes": 2 * GIB, "act_peak": 10 * GIB, "adapter_local_params": 2 * GIB // 12}
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
