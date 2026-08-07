from __future__ import annotations

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability

from miles.utils.workers.pod_context import (
    CELL_INDEX_ENV_VAR,
    POD_INDEX_ENV_VAR,
    SUBPROCESS_INDEX_ENV_VAR,
    read_pod_rank,
    read_rank_in_pod,
)
from miles.utils.workers.worker_spec import SchedulingSpec


def scheduling(*, ranks_per_pod: int = 1, gpu_slots_per_rank: int = 0, pods_per_cell: int = 1) -> SchedulingSpec:
    return SchedulingSpec(
        num_cells=1,
        num_workers_per_cell=ranks_per_pod * pods_per_cell,
        num_gpus_per_worker=gpu_slots_per_rank,
        num_gpu_slots_per_worker=gpu_slots_per_rank,
        num_gpus_per_node=ranks_per_pod * gpu_slots_per_rank,
    )


def _environ(*, cell_index: str = "0", **extra: str) -> dict[str, str]:
    return {CELL_INDEX_ENV_VAR: cell_index} | extra


class TestReadCellIndex:
    def test_reads_the_cell_the_platform_stamped_onto_the_pod(self):
        """The driver numbers the pool's cells the same way, so both sides must agree on this index."""
        assert read_pod_rank(scheduling=scheduling(), environ=_environ(cell_index="3")).cell_index == 3

    def test_refuses_a_pod_that_was_told_nothing_about_its_cell(self):
        """A pod that guessed would report itself as a member of whichever cell it invented."""
        with pytest.raises(AssertionError, match=CELL_INDEX_ENV_VAR):
            read_pod_rank(scheduling=scheduling(), environ={})


class TestReadPodRank:
    def test_numbers_the_ranks_of_a_multi_pod_cell_end_to_end(self):
        """A cell's ranks are numbered across its pods, so each pod has to offset by its own index."""
        rank = read_pod_rank(
            scheduling=scheduling(ranks_per_pod=8, gpu_slots_per_rank=1, pods_per_cell=2),
            environ={POD_INDEX_ENV_VAR: "2", SUBPROCESS_INDEX_ENV_VAR: "3"},
        )

        assert rank.worker_in_cell_index == 19

    def test_gives_each_rank_of_a_pod_its_own_gpus(self):
        """Two ranks that claimed the same gpu would each initialise a full model on it."""
        pod = scheduling(ranks_per_pod=4, gpu_slots_per_rank=2)
        first = read_pod_rank(scheduling=pod, environ=_environ(SUBPROCESS_INDEX_ENV_VAR="0"))
        second = read_pod_rank(scheduling=pod, environ=_environ(SUBPROCESS_INDEX_ENV_VAR="1"))

        assert (first.gpu_ids, second.gpu_ids) == ([0, 1], [2, 3])

    def test_a_single_rank_pod_needs_nothing_beyond_its_cell_index(self):
        """A static worker runs one rank with no supervisor, and must not depend on variables nobody sets."""
        rank = read_pod_rank(scheduling=scheduling(), environ=_environ())

        assert (rank.cell_index, rank.worker_in_cell_index, rank.gpu_ids) == (0, 0, [])

    def test_refuses_a_multi_pod_cell_whose_pod_was_told_no_index(self):
        """Every pod of the cell would then claim the leader's ranks, and the cell would have no others."""
        with pytest.raises(AssertionError, match=POD_INDEX_ENV_VAR):
            read_pod_rank(
                scheduling=scheduling(ranks_per_pod=4, gpu_slots_per_rank=1, pods_per_cell=2),
                environ=_environ(SUBPROCESS_INDEX_ENV_VAR="1"),
            )

    def test_the_ctor_context_carries_the_cell_the_pod_is_in(self):
        """ctor kwargs are computed from this context, and a trainer keys its checkpoints off the cell."""
        rank = read_pod_rank(
            scheduling=scheduling(ranks_per_pod=2, gpu_slots_per_rank=1),
            environ=_environ(SUBPROCESS_INDEX_ENV_VAR="1", cell_index="1"),
        )

        context = rank.ctor_context(capability=FakeBackendCapability())

        assert (context.cell_index, context.worker_in_cell_index, context.gpu_ids) == (1, 1, [1])

    def test_the_ctor_context_always_carries_a_provider_factory(self):
        """A spec asking for its engines must never find the factory missing, so it cannot be omitted."""
        rank = read_pod_rank(scheduling=scheduling(), environ=_environ())

        with pytest.raises(TypeError):
            rank.ctor_context()

    def test_refuses_a_subprocess_index_the_pod_was_not_launched_for(self):
        """A rank beyond the pod's share would collide with a rank of the next pod in the cell."""
        with pytest.raises(AssertionError, match=SUBPROCESS_INDEX_ENV_VAR):
            read_pod_rank(
                scheduling=scheduling(ranks_per_pod=2, gpu_slots_per_rank=1),
                environ=_environ(SUBPROCESS_INDEX_ENV_VAR="2"),
            )


class TestReadRankInPod:
    def test_reports_the_index_the_supervisor_gave_this_subprocess(self):
        """The rpc port is offset by this number, and only the supervisor's own variable carries it."""
        assert read_rank_in_pod({SUBPROCESS_INDEX_ENV_VAR: "3"}) == 3

    def test_an_unsupervised_process_is_the_only_rank_of_its_pod(self):
        """A worker launched without the supervisor must still bind the port the address book predicts."""
        assert read_rank_in_pod({}) == 0
