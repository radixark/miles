"""CPU unit tests for miles.utils.dp_schedule.build_dp_schedule.

The tests assert the invariants documented at the top of dp_schedule.py against
static / dynamic / VPP / oversize / balance / compact-rollout / trailing-trim
scenarios.

Trim, dynamic-gbs resolution, and per-rank rollout_data packaging all live in
``train_data_conversion.split_train_data_by_dp_scheduled_raw``; these tests just
exercise the schedule itself (``partitions`` and ``micro_batch_indices``).
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from miles.utils.dp_schedule import build_dp_schedule, has_full_schedule_config


def make_args(
    *,
    micro_batch_size=1,
    use_dynamic_batch_size=False,
    max_tokens_per_gpu=None,
    balance_data=False,
    balance_by_flops=False,
):
    return SimpleNamespace(
        micro_batch_size=micro_batch_size,
        use_dynamic_batch_size=use_dynamic_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        balance_data=balance_data,
        balance_by_flops=balance_by_flops,
        # Minimal model config for calculate_fwd_flops (balance_by_flops path).
        hidden_size=16,
        num_attention_heads=2,
        num_query_groups=2,
        vocab_size=32,
        ffn_hidden_size=64,
        num_experts=None,
        num_layers=2,
        kv_channels=8,
        q_lora_rank=None,  # no MLA in the stub config
        kv_lora_rank=None,
        qk_pos_emb_head_dim=0,
        qk_head_dim=8,
        v_head_dim=8,
    )


def make_tp(dp_size=1, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=None):
    return {
        "dp_size": dp_size,
        "cp_size": cp_size,
        "vpp_size": vpp_size,
        "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
    }


def assert_invariants(
    partitions,
    micro_batch_indices,
    num_microbatches,
    *,
    dp_size,
    expected_global_sample_indices,
    total_lengths,
    max_per_bin=None,
):
    """Check the invariants documented at the top of dp_schedule.py.

    ``expected_global_sample_indices`` is the set of global sample indices
    that should end up covered (after trim). Trailing rollouts that don't
    fit are excluded.
    """
    seen_global: set[int] = set()
    for r in range(dp_size):
        partition = partitions[r]
        mbi = micro_batch_indices[r]

        # Same micro-batch count per rank (PP sync).
        assert len(mbi) == sum(num_microbatches), f"rank {r}: micro-batch count mismatch"

        # Flattened micro_batch_indices == range(len(partition)).
        flat = [i for micro_batch in mbi for i in micro_batch]
        assert flat == list(range(len(partition))), f"rank {r}: micro_batch_indices don't tile [0, n)"

        # Disjoint partitions whose union covers every kept sample.
        assert seen_global.isdisjoint(partition), f"rank {r}: overlap with other ranks"
        seen_global.update(partition)
    assert seen_global == set(expected_global_sample_indices), "covered sample set mismatch"

    if max_per_bin is None:
        return

    # Every micro-batch <= max_per_bin tokens, EXCEPT a singleton bin holding an oversized sample.
    for r in range(dp_size):
        partition = partitions[r]
        for micro_batch in micro_batch_indices[r]:
            bin_total = sum(total_lengths[partition[i]] for i in micro_batch)
            if bin_total > max_per_bin:
                assert (
                    len(micro_batch) == 1
                ), f"rank {r}: micro-batch sum {bin_total} > {max_per_bin} but contains {len(micro_batch)} samples"


def test_static_stride_single_step():
    """Static + strided DP split, single step (1 rollout = 1 sample)."""
    total_lengths = [10] * 16
    rollout_indices = list(range(16))
    args = make_args(micro_batch_size=2)
    tp = make_tp(dp_size=4)

    partitions, mbi, nmb, num_rollouts_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=16, rollout_indices=rollout_indices
    )

    assert nmb == [2]
    assert num_rollouts_per_step == [16]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=4,
        expected_global_sample_indices=range(16),
        total_lengths=total_lengths,
    )


def test_static_balance_multi_step():
    """Static + balance_data + 2 training steps."""
    total_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1]
    rollout_indices = list(range(16))
    args = make_args(micro_batch_size=2, balance_data=True)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, num_rollouts_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, rollout_indices=rollout_indices
    )

    assert nmb == [2, 2]
    assert num_rollouts_per_step == [8, 8]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(16),
        total_lengths=total_lengths,
    )


def test_dynamic_uniform():
    """Dynamic micro-batching on uniform-length samples."""
    total_lengths = [5] * 8
    rollout_indices = list(range(8))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, num_rollouts_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, rollout_indices=rollout_indices
    )

    assert num_rollouts_per_step == [8]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(8),
        total_lengths=total_lengths,
        max_per_bin=10,
    )


def test_dynamic_oversized_sample_lands_alone():
    """A sample larger than max_per_bin must end up alone in its micro-batch."""
    total_lengths = [15, 3, 3, 3, 3, 3, 3, 3]
    rollout_indices = list(range(8))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, num_rollouts_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, rollout_indices=rollout_indices
    )

    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(8),
        total_lengths=total_lengths,
        max_per_bin=10,
    )
    oversize_idx = total_lengths.index(15)
    found = False
    for r in range(2):
        if oversize_idx not in partitions[r]:
            continue
        local = partitions[r].index(oversize_idx)
        for micro_batch in mbi[r]:
            if local in micro_batch:
                assert micro_batch == [local], f"oversized sample shares a micro-batch: {micro_batch}"
                found = True
    assert found


def test_dynamic_with_vpp_rounds_to_mb_group():
    """num_microbatches per rank should be a multiple of mb_group when vpp_size > 1."""
    total_lengths = [4] * 32
    rollout_indices = list(range(32))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=8)
    tp = make_tp(dp_size=2, vpp_size=2, microbatch_group_size_per_vp_stage=2)

    partitions, mbi, nmb, num_rollouts_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=16, rollout_indices=rollout_indices
    )

    for n in nmb:
        assert n % 2 == 0, f"num_microbatches {n} is not a multiple of mb_group=2"
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(32),
        total_lengths=total_lengths,
        max_per_bin=8,
    )


def test_rollout_grouping_keeps_samples_together():
    """compact / subagent simulation: rollout 0 emits 3 samples, rollout 1 emits 2,
    rollout 2 emits 4. Splitter keeps every rollout's samples in a single step."""
    rollout_indices = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    total_lengths = [3] * 9
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=1)

    partitions, mbi, nmb, num_rollouts_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=1, rollout_indices=rollout_indices
    )

    # 3 rollouts / 1 per step -> 3 steps, gbs constant.
    assert num_rollouts_per_step == [1, 1, 1]
    # For each step, collect the samples (global indices) that landed in that step's micro-batches
    # on rank 0, then verify they exactly equal the rollout's sample positions.
    expected_per_step = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
    rank0_partition = partitions[0]
    micro_batch_cursor = 0
    for step_i, n_micro_batches in enumerate(nmb):
        step_locals = sorted(
            j for micro_batch in mbi[0][micro_batch_cursor : micro_batch_cursor + n_micro_batches] for j in micro_batch
        )
        step_globals = [rank0_partition[j] for j in step_locals]
        assert (
            sorted(step_globals) == expected_per_step[step_i]
        ), f"step {step_i} samples = {step_globals}, expected {expected_per_step[step_i]}"
        micro_batch_cursor += n_micro_batches
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=1,
        expected_global_sample_indices=range(9),
        total_lengths=total_lengths,
        max_per_bin=12,
    )


def test_trims_trailing_rollouts_that_dont_fill_a_step():
    """5 rollouts, gbs=2 -> 2 steps x 2 rollouts; trailing rollout 4 (sample positions 6, 7)
    is dropped."""
    rollout_indices = [0, 0, 1, 2, 2, 3, 4, 4]
    total_lengths = [3] * 8
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=1)

    partitions, mbi, nmb, num_rollouts_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=2, rollout_indices=rollout_indices
    )

    assert num_rollouts_per_step == [2, 2]
    # Sample positions 6 and 7 belong to the trimmed rollout 4 and must be absent.
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=1,
        expected_global_sample_indices=range(6),
        total_lengths=total_lengths,
        max_per_bin=12,
    )


def test_partial_final_step_opt_in():
    """--allow-partial-train-step: 5 rollouts, gbs=2 -> 2 full steps + 1 partial
    step of the trailing rollout, covering every sample."""
    rollout_indices = [0, 0, 1, 2, 2, 3, 4, 4]
    total_lengths = [3] * 8
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    args.allow_partial_train_step = True
    tp = make_tp(dp_size=1)

    partitions, mbi, nmb, num_rollouts_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=2, rollout_indices=rollout_indices
    )

    assert num_rollouts_per_step == [2, 2, 1]
    assert len(nmb) == 3
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=1,
        expected_global_sample_indices=range(8),
        total_lengths=total_lengths,
        max_per_bin=12,
    )


def test_partial_final_step_skipped_when_below_dp_size():
    """A trailing rollout with fewer samples than dp_size cannot form a step."""
    rollout_indices = [0, 0, 1, 1, 2]
    total_lengths = [3] * 5
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    args.allow_partial_train_step = True
    tp = make_tp(dp_size=2)

    _, _, nmb, num_rollouts_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=2, rollout_indices=rollout_indices
    )

    assert num_rollouts_per_step == [2]
    assert len(nmb) == 1


def test_rejects_when_fewer_rollouts_than_gbs():
    """gbs=4 with only 3 distinct rollouts -> cannot form one step."""
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=1)
    with pytest.raises(AssertionError, match="global_batch_size"):
        build_dp_schedule(args, tp, [3] * 6, global_batch_size=4, rollout_indices=[0, 0, 1, 1, 2, 2])


def test_static_misaligned_micro_batch_count_asserts():
    """Static path: micro_batch count not a multiple of dp_size * mb_group must fail loudly."""
    args = make_args(micro_batch_size=3)
    tp = make_tp(dp_size=2)
    with pytest.raises(AssertionError, match="static path"):
        build_dp_schedule(args, tp, [10] * 8, global_batch_size=8, rollout_indices=list(range(8)))


def test_randomized_invariants_dynamic():
    """Randomized sweep of the documented invariants on the dynamic path,
    including random compact rollout groupings."""
    rng = random.Random(7)
    for _ in range(30):
        dp_size = rng.choice([1, 2, 4])
        gbs = dp_size * rng.choice([2, 4, 8])
        num_rollouts = gbs * rng.randint(1, 2) + rng.randint(0, 3)  # may leave a trailing partial step
        rollout_indices = []
        for rid in range(num_rollouts):
            rollout_indices.extend([rid] * rng.randint(1, 3))
        total_lengths = [rng.randint(1, 40) for _ in rollout_indices]
        max_tokens = rng.randint(20, 60)
        balance = rng.random() < 0.5

        args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=max_tokens, balance_data=balance)
        tp = make_tp(dp_size=dp_size)
        partitions, mbi, nmb, num_rollouts_per_step = build_dp_schedule(
            args, tp, total_lengths, global_batch_size=gbs, rollout_indices=rollout_indices
        )

        num_steps = num_rollouts // gbs
        assert num_rollouts_per_step == [gbs] * num_steps
        kept_rollouts = set(range(num_steps * gbs))
        expected = [i for i, rid in enumerate(rollout_indices) if rid in kept_rollouts]
        assert_invariants(
            partitions,
            mbi,
            nmb,
            dp_size=dp_size,
            expected_global_sample_indices=expected,
            total_lengths=total_lengths,
            max_per_bin=max_tokens,
        )


def test_balance_by_flops_packs_and_distributes():
    """balance_by_flops: KK-on-FLOPs packing, invariants preserved (the token cap
    is intentionally NOT enforced in this mode)."""
    total_lengths = [10, 200, 30, 400, 50, 600, 70, 800]
    rollout_indices = list(range(8))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=600, balance_by_flops=True)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, num_rollouts_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, rollout_indices=rollout_indices
    )

    assert num_rollouts_per_step == [8]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(8),
        total_lengths=total_lengths,
        max_per_bin=None,  # cap not enforced in FLOPs mode
    )


def test_balance_by_flops_singleton_fallback():
    """When ceil(total/cap) >= n samples, every sample gets its own micro_batch."""
    total_lengths = [100] * 4
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=100, balance_by_flops=True)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, _ = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=4, rollout_indices=list(range(4))
    )
    assert sum(nmb) * 2 == 4  # 4 singleton micro_batch over 2 ranks
    for r in range(2):
        for micro_batch in mbi[r]:
            assert len(micro_batch) == 1


def test_has_full_schedule_config():
    assert has_full_schedule_config(make_tp())
    assert not has_full_schedule_config({})
    assert not has_full_schedule_config(None)
    assert not has_full_schedule_config({"dp_size": 4})  # fsdp/torchtitan shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
