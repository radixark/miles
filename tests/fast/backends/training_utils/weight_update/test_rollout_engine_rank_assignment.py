"""`assign_rollout_engine_ranks` must give every rank of the rollout engines handed over exactly one writer."""

from collections import Counter
from functools import partial
from types import SimpleNamespace

import pytest
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.parallel import GroupInfo
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocols.shared_utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
    assign_rollout_engine_ranks,
    assign_rollout_engine_ranks_for_data_replica,
)


def _targets_of(assignments: list[RolloutEngineRankAssignment]) -> list[tuple[int, int]]:
    return [
        (rollout_engine_ind, assignment.rollout_engine_rank)
        for assignment in assignments
        for rollout_engine_ind in assignment.rollout_engine_indices
    ]


@pytest.mark.parametrize("data_replica_size", [1, 2, 3, 4, 6, 8, 16])
@pytest.mark.parametrize("engine_gpu_counts", [[2], [8], [4, 4], [2, 2, 2], [1, 3, 4], [8, 8, 8, 8], []])
def test_assignments_follow_engine_gpu_counts(data_replica_size: int, engine_gpu_counts: list[int]):
    """Writers cover exactly the rollout engine ranks in the counts: none invented, none skipped, none written
    twice."""
    writers = Counter(
        target
        for data_replica_rank in range(data_replica_size)
        for target in _targets_of(
            assign_rollout_engine_ranks_for_data_replica(data_replica_rank, data_replica_size, engine_gpu_counts)
        )
    )

    expected = {
        (rollout_engine_ind, rollout_engine_rank)
        for rollout_engine_ind, gpu_count in enumerate(engine_gpu_counts)
        for rollout_engine_rank in range(gpu_count)
    }
    assert set(writers) == expected
    assert set(writers.values()) <= {1}


@pytest.mark.parametrize("gpu_count", [0, -1])
def test_a_rollout_engine_without_gpus_is_rejected(gpu_count: int):
    """A rollout engine with no GPUs fails loudly instead of being silently left out."""
    with pytest.raises(AssertionError, match="rollout engine 1 has"):
        assign_rollout_engine_ranks_for_data_replica(
            data_replica_rank=0, data_replica_size=2, engine_gpu_counts=[2, gpu_count]
        )


def test_data_replicas_beyond_the_rollout_engine_ranks_are_not_senders():
    """A data replica with no rollout engine rank must report itself as no sender rather than query an engine."""
    assert (
        assign_rollout_engine_ranks_for_data_replica(data_replica_rank=2, data_replica_size=8, engine_gpu_counts=[2])
        == []
    )


def test_extra_engines_reuse_a_data_replica_already_on_their_rollout_engine_rank():
    """A data replica that already sends a rollout engine rank's weights serves the same rank of the next engine."""
    assignments = [
        assign_rollout_engine_ranks_for_data_replica(data_replica_rank, 4, [3, 3]) for data_replica_rank in range(4)
    ]

    assert assignments == [
        [RolloutEngineRankAssignment(rollout_engine_rank=0, rollout_engine_indices=(0,))],
        [RolloutEngineRankAssignment(rollout_engine_rank=1, rollout_engine_indices=(0, 1))],
        [RolloutEngineRankAssignment(rollout_engine_rank=2, rollout_engine_indices=(0, 1))],
        [RolloutEngineRankAssignment(rollout_engine_rank=0, rollout_engine_indices=(1,))],
    ]


_WORLD_SIZE = 4
_PP_SIZE = 2
_ENGINE_GPU_COUNTS = [2, 2]


def _assign_on_every_rank(rank: int, world_size: int, port: int, *, gather_pp: bool) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        stride = world_size // _PP_SIZE
        pp_groups = [dist.new_group(list(range(column, world_size, stride))) for column in range(stride)]
        pp_rank = rank // stride
        parallel_state = SimpleNamespace(pp=GroupInfo(rank=pp_rank, size=_PP_SIZE, group=pp_groups[rank % stride]))

        assignments = assign_rollout_engine_ranks(
            parallel_state, WeightUpdatePlacement(gather_pp=gather_pp), _ENGINE_GPU_COUNTS
        )

        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, (pp_rank, _targets_of(assignments)))
        if rank == 0:
            _assert_one_writer_per_rollout_engine_rank(gathered, gather_pp=gather_pp)
    finally:
        dist.destroy_process_group()


def _assert_one_writer_per_rollout_engine_rank(gathered: list, *, gather_pp: bool) -> None:
    all_targets = {
        (rollout_engine_ind, rollout_engine_rank)
        for rollout_engine_ind, gpu_count in enumerate(_ENGINE_GPU_COUNTS)
        for rollout_engine_rank in range(gpu_count)
    }
    if gather_pp:
        # every rank holds the whole model: one writer in the whole world
        writers = Counter(target for _pp_rank, targets in gathered for target in targets)
        assert set(writers) == all_targets and set(writers.values()) == {1}
    else:
        # every rank holds its PP stage: one writer per stage
        for pp_rank in range(_PP_SIZE):
            writers = Counter(target for stage, targets in gathered if stage == pp_rank for target in targets)
            assert set(writers) == all_targets and set(writers.values()) == {1}


def test_bridge_writes_each_rollout_engine_rank_once():
    """Megatron Bridge gathers PP, so every rank holds the whole model; planning per PP stage would make
    every stage write every rollout engine rank."""
    run_multiprocess(partial(_assign_on_every_rank, gather_pp=True), world_size=_WORLD_SIZE)


def test_direct_writes_each_rollout_engine_rank_once_per_pp_stage():
    """Megatron Direct keeps PP local, so each stage must reach every rollout engine rank with its own params."""
    run_multiprocess(partial(_assign_on_every_rank, gather_pp=False), world_size=_WORLD_SIZE)
