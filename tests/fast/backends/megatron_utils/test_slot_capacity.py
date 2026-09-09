"""Slot-capacity resolution: the flag-derived byte model and the min-of-the-
binding-constraints arithmetic; the GPU probe itself is exercised on hardware."""

from argparse import Namespace

import pytest

from miles.backends.megatron_utils.lora.slot_capacity import RankProbe, bytes_per_train_param, resolve_slot_capacity

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


def _probe(free_before=100 * GIB, slot=2 * GIB, act_peak=10 * GIB, full_params=GIB) -> RankProbe:
    return RankProbe(
        free_before=free_before,
        free_after=free_before - slot,
        act_peak=act_peak,
        adapter_local_params=full_params // 4,
        adapter_full_params=full_params,
    )


def _args(**overrides) -> Namespace:
    defaults = dict(lora_rank=32, train_memory_margin_bytes=2 * GIB, engine_host_lora_budget_bytes=None)
    return Namespace(**{**defaults, **overrides})


def test_the_worst_rank_bounds_the_capacity():
    probes = [_probe(), _probe(free_before=50 * GIB)]
    # worst rank: (50 - 10 - 2) / 2 = 19 slots
    assert resolve_slot_capacity(_args(), probes, keep_k=2) == 19


def test_host_budget_binds_when_smaller():
    # gpu allows (100-10-2)/2 = 44; host allows 12GiB / (2 * 2GiB/version) = 3
    args = _args(engine_host_lora_budget_bytes=12 * GIB)
    assert resolve_slot_capacity(args, [_probe()], keep_k=2) == 3


def test_no_room_for_one_slot_is_a_launch_error():
    probes = [_probe(free_before=13 * GIB, slot=2 * GIB, act_peak=10 * GIB)]  # (13-10-2)/2 = 0
    with pytest.raises(AssertionError, match="Lower --lora-rank"):
        resolve_slot_capacity(_args(), probes, keep_k=2)


def test_capacity_never_goes_negative():
    assert _probe(free_before=5 * GIB, act_peak=10 * GIB).capacity(margin_bytes=0) == 0
