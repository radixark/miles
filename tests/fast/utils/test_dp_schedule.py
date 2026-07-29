"""CPU unit tests for miles.utils.dp_schedule.build_dp_schedule.

The tests assert the invariants documented at the top of dp_schedule.py against
a range of static / dynamic / VPP / oversize / balance scenarios, plus an
equivalence check of ``num_microbatches`` against the legacy train-side
computation (``get_minimum_num_micro_batch_size`` + DP-wide MAX) that this
schedule replaces.

Trim, dynamic-gbs resolution, and per-rank rollout_data packaging all live in
``train_data_conversion.split_train_data_by_dp_scheduled``; these tests just
exercise the schedule itself (``partitions`` and ``micro_batch_indices``).
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from miles.utils.data import get_minimum_num_micro_batch_size
from miles.utils.dp_schedule import build_dp_schedule, has_full_schedule_config


def make_args(
    *,
    micro_batch_size=1,
    use_dynamic_batch_size=False,
    max_tokens_per_gpu=None,
    balance_data=False,
):
    return SimpleNamespace(
        micro_batch_size=micro_batch_size,
        use_dynamic_batch_size=use_dynamic_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        balance_data=balance_data,
    )


def make_tp(dp_size=1, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=None):
    return {
        "dp_size": dp_size,
        "cp_size": cp_size,
        "vpp_size": vpp_size,
        "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
    }


def assert_invariants(partitions, micro_batch_indices, num_microbatches, *, dp_size, total_lengths, max_per_bin=None):
    """Check the invariants documented at the top of dp_schedule.py."""
    expected_per_rank = len(total_lengths) // dp_size
    seen_global: set[int] = set()
    for r in range(dp_size):
        partition = partitions[r]
        mbi = micro_batch_indices[r]

        # Same sample count per rank.
        assert len(partition) == expected_per_rank, f"rank {r}: {len(partition)} samples, want {expected_per_rank}"

        # Same num_mbs per rank (PP sync); num_microbatches is shared, so each rank's
        # flat mbs count must match sum(num_microbatches).
        assert len(mbi) == sum(num_microbatches), f"rank {r}: mbs count mismatch"

        # Flattened micro_batch_indices == range(len(partition)) (each sample covered
        # exactly once, by exactly one mbs).
        flat = [i for mbs in mbi for i in mbs]
        assert flat == list(range(len(partition))), f"rank {r}: micro_batch_indices don't tile [0, n)"

        # Disjoint partitions whose union covers every sample.
        assert seen_global.isdisjoint(partition), f"rank {r}: overlap with other ranks"
        seen_global.update(partition)
    assert seen_global == set(range(len(total_lengths))), "some samples not assigned to any rank"

    if max_per_bin is None:
        return

    # Every mbs <= max_per_bin tokens, EXCEPT a singleton bin holding an oversized sample.
    for r in range(dp_size):
        partition = partitions[r]
        for mbs in micro_batch_indices[r]:
            bin_total = sum(total_lengths[partition[i]] for i in mbs)
            if bin_total > max_per_bin:
                assert len(mbs) == 1, f"rank {r}: mbs sum {bin_total} > {max_per_bin} but contains {len(mbs)} samples"


def test_static_stride_single_step():
    """Static + strided DP split, single step."""
    total_lengths = [10] * 16
    args = make_args(micro_batch_size=2)
    tp = make_tp(dp_size=4)

    partitions, mbi, nmb = build_dp_schedule(args, tp, total_lengths, global_batch_size=16)

    assert nmb == [2]
    assert_invariants(partitions, mbi, nmb, dp_size=4, total_lengths=total_lengths)


def test_static_stride_matches_legacy_partition_order():
    """Static + strided must produce exactly the legacy strided row order —
    rank r gets rows [r, r+dp, r+2*dp, ...] in that order, contiguously chunked."""
    total_lengths = list(range(100, 116))
    args = make_args(micro_batch_size=2)
    tp = make_tp(dp_size=4)

    partitions, _, _ = build_dp_schedule(args, tp, total_lengths, global_batch_size=16)

    for r in range(4):
        assert partitions[r] == list(range(r, 16, 4)), f"rank {r} partition deviates from legacy strided order"


def test_static_balance_multi_step():
    """Static + balance_data + 2 training steps. Each rank must get gbs/dp per step."""
    total_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1]  # 2 steps of 8
    args = make_args(micro_batch_size=2, balance_data=True)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb = build_dp_schedule(args, tp, total_lengths, global_batch_size=8)

    assert nmb == [2, 2]
    assert_invariants(partitions, mbi, nmb, dp_size=2, total_lengths=total_lengths)


def test_dynamic_uniform():
    """Dynamic mbs on uniform-length samples."""
    total_lengths = [5] * 8
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb = build_dp_schedule(args, tp, total_lengths, global_batch_size=8)

    assert_invariants(partitions, mbi, nmb, dp_size=2, total_lengths=total_lengths, max_per_bin=10)


def test_dynamic_skewed_lengths():
    """Skewed lengths (the case where K-K used to over-pack a single bin)."""
    total_lengths = [9, 9, 9, 9, 1, 1, 1, 1]
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb = build_dp_schedule(args, tp, total_lengths, global_batch_size=8)

    assert_invariants(partitions, mbi, nmb, dp_size=2, total_lengths=total_lengths, max_per_bin=10)


def test_dynamic_oversized_sample_lands_alone():
    """A single sample exceeding max_per_bin must end up alone in its mbs (with no
    other samples crammed in)."""
    total_lengths = [15, 3, 3, 3, 3, 3, 3, 3]  # 15 > C=10
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb = build_dp_schedule(args, tp, total_lengths, global_batch_size=8)

    assert_invariants(partitions, mbi, nmb, dp_size=2, total_lengths=total_lengths, max_per_bin=10)
    # Find the rank holding the oversized sample and verify it lives alone in some mbs.
    oversize_idx = total_lengths.index(15)
    found = False
    for r in range(2):
        partition = partitions[r]
        if oversize_idx not in partition:
            continue
        local = partition.index(oversize_idx)
        for mbs in mbi[r]:
            if local in mbs:
                assert mbs == [local], f"oversized sample shares an mbs: {mbs}"
                found = True
    assert found


def test_dynamic_with_vpp_rounds_to_mb_group():
    """num_microbatches per rank should be a multiple of mb_group when vpp_size > 1."""
    total_lengths = [4] * 32  # 2 steps of 16; per step, ~8 bins of 8 needed at C=8
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=8)
    tp = make_tp(dp_size=2, vpp_size=2, microbatch_group_size_per_vp_stage=2)

    partitions, mbi, nmb = build_dp_schedule(args, tp, total_lengths, global_batch_size=16)

    for n in nmb:
        assert n % 2 == 0, f"num_microbatches {n} is not a multiple of mb_group=2"
    assert_invariants(partitions, mbi, nmb, dp_size=2, total_lengths=total_lengths, max_per_bin=8)


def test_static_indivisible_per_rank_batch_asserts():
    """gbs/dp not a multiple of micro_batch_size must fail loudly, not mis-schedule."""
    args = make_args(micro_batch_size=3)
    tp = make_tp(dp_size=2)
    with pytest.raises(AssertionError, match="micro_batch_size"):
        build_dp_schedule(args, tp, [10] * 8, global_batch_size=8)


def test_untrimmed_input_asserts():
    """num_samples not a multiple of global_batch_size must fail loudly."""
    args = make_args(micro_batch_size=1)
    tp = make_tp(dp_size=2)
    with pytest.raises(AssertionError, match="multiple of global_batch_size"):
        build_dp_schedule(args, tp, [10] * 10, global_batch_size=8)


def test_dynamic_num_microbatches_matches_legacy_train_side():
    """The PP-sync-critical value: for the same strided partition, the rollout-side
    schedule must produce exactly the num_microbatches the legacy train-side path
    computed (per-rank ``get_minimum_num_micro_batch_size``, then DP-wide MAX)."""
    rng = random.Random(42)
    dp_size, cp_size = 4, 2
    max_tokens_per_gpu = 512
    global_batch_size = 32

    for _ in range(50):
        num_steps = rng.randint(1, 3)
        total_lengths = [rng.randint(16, 1200) for _ in range(global_batch_size * num_steps)]

        args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=max_tokens_per_gpu)
        tp = make_tp(dp_size=dp_size, cp_size=cp_size)
        _, _, nmb = build_dp_schedule(args, tp, total_lengths, global_batch_size=global_batch_size)

        # Legacy: each rank r holds the strided rows of each step, computes its own
        # first-fit bin count over its local per-step slice, and the DP group takes MAX.
        num_local_gbs = global_batch_size // dp_size
        legacy_nmb = []
        for step_i in range(num_steps):
            step = total_lengths[step_i * global_batch_size : (step_i + 1) * global_batch_size]
            per_rank = []
            for r in range(dp_size):
                local = [step[i] for i in range(r, global_batch_size, dp_size)]
                assert len(local) == num_local_gbs
                per_rank.append(get_minimum_num_micro_batch_size(local, max_tokens_per_gpu * cp_size))
            legacy_nmb.append(max(per_rank))

        assert nmb == legacy_nmb, f"num_microbatches diverged from legacy: {nmb} vs {legacy_nmb}"


def test_randomized_invariants_dynamic():
    """Randomized sweep of the documented invariants on the dynamic path."""
    rng = random.Random(7)
    for _ in range(30):
        dp_size = rng.choice([1, 2, 4])
        gbs = dp_size * rng.choice([2, 4, 8])
        num_steps = rng.randint(1, 2)
        total_lengths = [rng.randint(1, 40) for _ in range(gbs * num_steps)]
        max_tokens = rng.randint(20, 60)
        balance = rng.random() < 0.5

        args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=max_tokens, balance_data=balance)
        tp = make_tp(dp_size=dp_size)
        partitions, mbi, nmb = build_dp_schedule(args, tp, total_lengths, global_batch_size=gbs)

        assert_invariants(
            partitions, mbi, nmb, dp_size=dp_size, total_lengths=total_lengths, max_per_bin=max_tokens
        )


def test_has_full_schedule_config():
    assert has_full_schedule_config(make_tp())
    assert not has_full_schedule_config({})
    assert not has_full_schedule_config(None)
    assert not has_full_schedule_config({"dp_size": 4})  # fsdp/torchtitan shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
