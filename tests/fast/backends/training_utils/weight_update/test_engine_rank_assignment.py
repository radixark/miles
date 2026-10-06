"""`assign_engine_ranks` must give every engine rank of the engines handed over exactly one writer."""

from collections import Counter
from functools import partial
from types import SimpleNamespace

import pytest
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.parallel import GroupInfo
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocols.shared_utils.engine_rank_assignment import (
    EngineRankAssignment,
    assign_engine_ranks,
    assign_engine_ranks_for_replica,
)


def _targets_of(assignments: list[EngineRankAssignment]) -> list[tuple[int, int]]:
    return [
        (engine_index, assignment.engine_rank)
        for assignment in assignments
        for engine_index in assignment.engine_indices
    ]


@pytest.mark.parametrize("replica_size", [1, 2, 3, 4, 6, 8, 16])
@pytest.mark.parametrize("engine_gpu_counts", [[2], [8], [4, 4], [2, 2, 2], [1, 3, 4], [8, 8, 8, 8], []])
def test_assignments_follow_engine_gpu_counts(replica_size: int, engine_gpu_counts: list[int]):
    """Writers cover exactly the engine ranks in the counts: none invented, none skipped, none written twice."""
    writers = Counter(
        target
        for replica_rank in range(replica_size)
        for target in _targets_of(assign_engine_ranks_for_replica(replica_rank, replica_size, engine_gpu_counts))
    )

    expected = {
        (engine_index, engine_rank)
        for engine_index, gpu_count in enumerate(engine_gpu_counts)
        for engine_rank in range(gpu_count)
    }
    assert set(writers) == expected
    assert set(writers.values()) <= {1}


def test_replicas_beyond_the_engine_ranks_are_not_senders():
    """A replica with no engine rank must report itself as no sender rather than query an engine."""
    assert assign_engine_ranks_for_replica(replica_rank=2, replica_size=8, engine_gpu_counts=[2]) == []


def test_extra_engines_reuse_a_replica_already_on_their_engine_rank():
    """A replica that already loads an engine rank's shard serves the same rank of the next engine."""
    assignments = [assign_engine_ranks_for_replica(replica_rank, 4, [3, 3]) for replica_rank in range(4)]

    assert assignments == [
        [EngineRankAssignment(engine_rank=0, engine_indices=(0,))],
        [EngineRankAssignment(engine_rank=1, engine_indices=(0, 1))],
        [EngineRankAssignment(engine_rank=2, engine_indices=(0, 1))],
        [EngineRankAssignment(engine_rank=0, engine_indices=(1,))],
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

        assignments = assign_engine_ranks(
            parallel_state, WeightUpdatePlacement(gather_pp=gather_pp), _ENGINE_GPU_COUNTS
        )

        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, (pp_rank, _targets_of(assignments)))
        if rank == 0:
            _assert_one_writer_per_engine_rank(gathered, gather_pp=gather_pp)
    finally:
        dist.destroy_process_group()


def _assert_one_writer_per_engine_rank(gathered: list, *, gather_pp: bool) -> None:
    all_targets = {
        (engine_index, engine_rank)
        for engine_index, gpu_count in enumerate(_ENGINE_GPU_COUNTS)
        for engine_rank in range(gpu_count)
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


def test_bridge_writes_each_engine_rank_once():
    """Megatron Bridge gathers PP, so every rank holds the whole model; planning per PP stage would make
    every stage write every engine rank."""
    run_multiprocess(partial(_assign_on_every_rank, gather_pp=True), world_size=_WORLD_SIZE)


def test_direct_writes_each_engine_rank_once_per_pp_stage():
    """Megatron Direct keeps PP local, so each stage must reach every engine rank with its own params."""
    run_multiprocess(partial(_assign_on_every_rank, gather_pp=False), world_size=_WORLD_SIZE)
