from argparse import Namespace

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.cp_utils import slice_log_prob_with_cp
from miles.backends.training_utils.loss_hub.advantages import normalize_advantages
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _reference_whiten(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Single-process mirror of `distributed_masked_whiten`: masked mean and
    variance, Bessel correction, and the same epsilon."""
    count = mask.sum()
    mean = (values * mask).sum() / count
    var = ((values**2) * mask).sum() / count - mean**2
    if count >= 2:
        var = var * count / (count - 1)
    return (values - mean) * torch.rsqrt(var + 1e-8)


def _dp_cp_parallel_state(rank: int, dp_size: int, cp_size: int, *, independent: bool = False) -> ParallelState:
    """ParallelState over a DP×CP test world of `dp_size * cp_size` gloo ranks.

    `independent=False` models the INTRA mode: `intra_dp_cp` is the DP×CP union,
    i.e. the whole world. `independent=True` models the INDEP mode: the
    Megatron-side DP is trivial, the CP groups are the inner level and the DP
    replicas the outer level. Subgroup creation is collective, so every rank
    must build every group in the same deterministic order.
    """
    dp_rank, cp_rank = divmod(rank, cp_size)

    dp_groups = [dist.new_group([d * cp_size + c for d in range(dp_size)]) for c in range(cp_size)]
    cp_groups = [dist.new_group([d * cp_size + c for c in range(cp_size)]) for d in range(dp_size)] if cp_size > 1 else []
    trivial = GroupInfo(rank=0, size=1, group=None)

    if independent:
        intra_dp_cp = GroupInfo(rank=cp_rank, size=cp_size, group=cp_groups[dp_rank])
        indep_dp = GroupInfo(rank=dp_rank, size=dp_size, group=dp_groups[cp_rank])
        intra_dp = trivial
    else:
        intra_dp_cp = GroupInfo(rank=rank, size=dp_size * cp_size, group=dist.group.WORLD)
        intra_dp = GroupInfo(rank=dp_rank, size=dp_size, group=dp_groups[cp_rank])
        indep_dp = trivial

    return ParallelState(
        intra_dp=intra_dp,
        intra_dp_cp=intra_dp_cp,
        cp=GroupInfo(rank=cp_rank, size=cp_size, group=cp_groups[dp_rank] if cp_size > 1 else None),
        tp=trivial,
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=indep_dp,
    )


def _run_normalize_case(
    rank: int,
    per_replica: list[tuple[int, int, torch.Tensor, torch.Tensor]],
    dp_size: int,
    cp_size: int,
    *,
    independent: bool = False,
) -> list[torch.Tensor]:
    """Whiten this rank's CP shard of its DP replica's single-sample batch.

    `per_replica` holds one `(total_length, response_length, values, mask)`
    tuple per DP replica; every rank knows the whole logical batch so it can
    also derive the expected output locally.
    """
    set_parallel_state(_dp_cp_parallel_state(rank, dp_size, cp_size, independent=independent))
    dp_rank = rank // cp_size
    total_length, response_length, values, mask = per_replica[dp_rank]

    args = Namespace(qkv_format="thd")
    local_advs = [slice_log_prob_with_cp(values, total_length, response_length)]
    return normalize_advantages(args, local_advs, [mask], [total_length], [response_length])


def _expected_local_shard(
    rank: int,
    per_replica: list[tuple[int, int, torch.Tensor, torch.Tensor]],
    cp_size: int,
) -> torch.Tensor:
    """Whiten the full logical batch on one process, then take this rank's shard."""
    all_values = torch.cat([values for _, _, values, _ in per_replica])
    all_masks = torch.cat([mask for _, _, _, mask in per_replica])
    whitened = _reference_whiten(all_values, all_masks)

    response_lengths = [response_length for _, response_length, _, _ in per_replica]
    dp_rank = rank // cp_size
    start = sum(response_lengths[:dp_rank])
    total_length, response_length, _, _ = per_replica[dp_rank]
    return slice_log_prob_with_cp(whitened[start : start + response_length], total_length, response_length)


def _worker_whitens_over_cp_and_dp(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        per_replica = [
            (8, 6, torch.tensor([3.0, -1.0, 2.5, 0.5, -2.0, 4.0]), torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])),
            (7, 5, torch.tensor([-3.5, 1.5, 6.0, -0.5, 2.0]), torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])),
        ]
        whitened = _run_normalize_case(rank, per_replica, dp_size=2, cp_size=2)
        torch.testing.assert_close(whitened[0], _expected_local_shard(rank, per_replica, cp_size=2))
    finally:
        dist.destroy_process_group()


def _worker_empty_shard_joins_collective(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        # The zigzag split of total=3, response=1 places replica 0's only
        # token on cp rank 1, leaving rank 0 with an empty shard. Skipping the
        # collective there leaves rank 2 waiting inside the all_reduce of
        # their shared group; completing this worker at all is the regression
        # check.
        per_replica = [
            (3, 1, torch.tensor([2.0]), torch.tensor([1.0])),
            (9, 7, torch.tensor([5.0, -4.0, 3.5, -1.0, 2.0, -3.0, 1.5]), torch.ones(7)),
        ]
        whitened = _run_normalize_case(rank, per_replica, dp_size=2, cp_size=2)
        expected = _expected_local_shard(rank, per_replica, cp_size=2)
        assert whitened[0].numel() == expected.numel()
        if whitened[0].numel():
            torch.testing.assert_close(whitened[0], expected)
    finally:
        dist.destroy_process_group()


def _worker_zero_mask_cp_slice_does_not_raise(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        # Every token stored on cp rank 0 is masked out on both replicas, but
        # the global batch still has unmasked tokens on cp rank 1. Judging the
        # zero-mask error from one CP slice's group raises on a valid batch.
        per_replica = [
            (8, 6, torch.tensor([3.0, -1.0, 2.5, 0.5, -2.0, 4.0]), torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 0.0])),
            (9, 7, torch.tensor([5.0, -4.0, 3.5, -1.0, 2.0, -3.0, 1.5]), torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])),
        ]
        whitened = _run_normalize_case(rank, per_replica, dp_size=2, cp_size=2)
        expected = _expected_local_shard(rank, per_replica, cp_size=2)
        assert whitened[0].numel() == expected.numel()
        torch.testing.assert_close(whitened[0], expected)
    finally:
        dist.destroy_process_group()


def _worker_global_zero_mask_raises_on_every_rank(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        per_replica = [
            (8, 6, torch.tensor([3.0, -1.0, 2.5, 0.5, -2.0, 4.0]), torch.zeros(6)),
            (9, 7, torch.tensor([5.0, -4.0, 3.5, -1.0, 2.0, -3.0, 1.5]), torch.zeros(7)),
        ]
        with pytest.raises(ValueError, match="mask sum"):
            _run_normalize_case(rank, per_replica, dp_size=2, cp_size=2)
    finally:
        dist.destroy_process_group()


def _worker_cp1_dp2_matches_reference(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        per_replica = [
            (6, 4, torch.tensor([3.0, -1.0, 2.5, 0.5]), torch.tensor([1.0, 1.0, 0.0, 1.0])),
            (5, 3, torch.tensor([-3.5, 1.5, 6.0]), torch.tensor([1.0, 1.0, 1.0])),
        ]
        whitened = _run_normalize_case(rank, per_replica, dp_size=2, cp_size=1)
        torch.testing.assert_close(whitened[0], _expected_local_shard(rank, per_replica, cp_size=1))
    finally:
        dist.destroy_process_group()


def _worker_independent_dp_whitens_over_cp_and_replicas(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        per_replica = [
            (8, 6, torch.tensor([3.0, -1.0, 2.5, 0.5, -2.0, 4.0]), torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])),
            (7, 5, torch.tensor([-3.5, 1.5, 6.0, -0.5, 2.0]), torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])),
        ]
        whitened = _run_normalize_case(rank, per_replica, dp_size=2, cp_size=2, independent=True)
        torch.testing.assert_close(whitened[0], _expected_local_shard(rank, per_replica, cp_size=2))
    finally:
        dist.destroy_process_group()


def test_whitening_statistics_span_cp_and_dp_shards() -> None:
    run_multiprocess(_worker_whitens_over_cp_and_dp, world_size=4)


def test_empty_cp_shard_joins_the_collective() -> None:
    run_multiprocess(_worker_empty_shard_joins_collective, world_size=4)


def test_zero_mask_cp_slice_does_not_raise_on_valid_global_batch() -> None:
    run_multiprocess(_worker_zero_mask_cp_slice_does_not_raise, world_size=4)


def test_global_zero_mask_raises_on_every_rank() -> None:
    run_multiprocess(_worker_global_zero_mask_raises_on_every_rank, world_size=4)


def test_cp1_dp2_whitening_matches_full_batch_reference() -> None:
    run_multiprocess(_worker_cp1_dp2_matches_reference, world_size=2)


def test_independent_dp_whitens_over_cp_and_replicas() -> None:
    run_multiprocess(_worker_independent_dp_whitens_over_cp_and_replicas, world_size=4)
