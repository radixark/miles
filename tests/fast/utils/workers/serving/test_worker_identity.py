from __future__ import annotations

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability

from miles.utils.workers.serving.worker_identity import (
    CELL_INDEX_ENV_VAR,
    POD_INDEX_ENV_VAR,
    SUBPROCESS_INDEX_ENV_VAR,
    read_worker_identity,
    read_worker_in_pod_index,
)
from miles.utils.workers.worker_spec import SchedulingSpec


def scheduling(*, workers_per_pod: int = 1, gpu_slots_per_worker: int = 0, pods_per_cell: int = 1) -> SchedulingSpec:
    return SchedulingSpec(
        num_cells=1,
        num_workers_per_cell=workers_per_pod * pods_per_cell,
        num_gpus_per_worker=gpu_slots_per_worker,
        num_gpu_slots_per_worker=gpu_slots_per_worker,
        num_gpus_per_node=workers_per_pod * gpu_slots_per_worker,
    )


def _environ(*, cell_index: str = "0", pod_index: str | None = None, subprocess_index: str | None = None):
    environ = {CELL_INDEX_ENV_VAR: cell_index}
    if pod_index is not None:
        environ[POD_INDEX_ENV_VAR] = pod_index
    if subprocess_index is not None:
        environ[SUBPROCESS_INDEX_ENV_VAR] = subprocess_index
    return environ


class TestEnvVarOwnership:
    def test_every_index_a_pod_reads_is_named_by_miles(self):
        """A pod created by something other than our chart cannot be asked to set an upstream name."""
        assert (CELL_INDEX_ENV_VAR, POD_INDEX_ENV_VAR) == ("MILES_CELL_INDEX", "MILES_POD_INDEX")


class TestReadCellIndex:
    def test_reads_the_cell_the_platform_stamped_onto_the_pod(self):
        """The driver numbers the pool's cells the same way, so both sides must agree on this index."""
        assert read_worker_identity(scheduling=scheduling(), environ=_environ(cell_index="3")).cell_index == 3

    def test_refuses_a_pod_that_was_told_nothing_about_its_cell(self):
        """A pod that guessed would report itself as a member of whichever cell it invented."""
        with pytest.raises(AssertionError, match=CELL_INDEX_ENV_VAR):
            read_worker_identity(scheduling=scheduling(), environ={})


class TestReadWorkerIdentity:
    def test_numbers_the_workers_of_a_multi_pod_cell_end_to_end(self):
        """A cell's workers are numbered across its pods, so each pod has to offset by its own index."""
        identity = read_worker_identity(
            scheduling=scheduling(workers_per_pod=8, gpu_slots_per_worker=1, pods_per_cell=2),
            environ=_environ(pod_index="1", subprocess_index="3"),
        )

        assert identity.worker_in_cell_index == 11

    def test_refuses_a_pod_index_the_cell_does_not_have(self):
        """A pod numbered past the cell would report workers that belong to no cell of the pool."""
        with pytest.raises(AssertionError, match=POD_INDEX_ENV_VAR):
            read_worker_identity(
                scheduling=scheduling(workers_per_pod=8, gpu_slots_per_worker=1, pods_per_cell=2),
                environ=_environ(pod_index="2", subprocess_index="0"),
            )

    def test_gives_each_worker_of_a_pod_its_own_gpus(self):
        """Two workers that claimed the same gpu would each initialise a full model on it."""
        pod = scheduling(workers_per_pod=4, gpu_slots_per_worker=2)
        first = read_worker_identity(scheduling=pod, environ=_environ(subprocess_index="0"))
        second = read_worker_identity(scheduling=pod, environ=_environ(subprocess_index="1"))

        assert (first.gpu_ids, second.gpu_ids) == ([0, 1], [2, 3])

    def test_a_single_worker_pod_needs_nothing_beyond_its_cell_index(self):
        """A static worker runs one worker with no supervisor, and must not depend on variables nobody sets."""
        identity = read_worker_identity(scheduling=scheduling(), environ=_environ())

        assert (identity.cell_index, identity.worker_in_cell_index, identity.gpu_ids) == (0, 0, [])

    def test_refuses_a_multi_pod_cell_whose_pod_was_told_no_index(self):
        """Every pod of the cell would then claim the leader's workers, and the cell would have no others."""
        with pytest.raises(AssertionError, match=POD_INDEX_ENV_VAR):
            read_worker_identity(
                scheduling=scheduling(workers_per_pod=4, gpu_slots_per_worker=1, pods_per_cell=2),
                environ=_environ(subprocess_index="1"),
            )

    def test_the_ctor_context_carries_the_cell_the_pod_is_in(self):
        """ctor kwargs are computed from this context, and a trainer keys its checkpoints off the cell."""
        identity = read_worker_identity(
            scheduling=scheduling(workers_per_pod=2, gpu_slots_per_worker=1),
            environ=_environ(subprocess_index="1", cell_index="1"),
        )

        context = identity.ctor_context(capability=FakeBackendCapability())

        assert (context.cell_index, context.worker_in_cell_index, context.gpu_ids) == (1, 1, [1])

    def test_the_ctor_context_always_carries_a_provider_factory(self):
        """A spec asking for its engines must never find the factory missing, so it cannot be omitted."""
        identity = read_worker_identity(scheduling=scheduling(), environ=_environ())

        with pytest.raises(TypeError):
            identity.ctor_context()

    def test_refuses_a_subprocess_index_the_pod_was_not_launched_for(self):
        """A worker beyond the pod's share would collide with a worker of the next pod in the cell."""
        with pytest.raises(AssertionError, match=SUBPROCESS_INDEX_ENV_VAR):
            read_worker_identity(
                scheduling=scheduling(workers_per_pod=2, gpu_slots_per_worker=1),
                environ=_environ(subprocess_index="2"),
            )


class TestReadWorkerInPodIndex:
    def test_reports_the_index_the_supervisor_gave_this_subprocess(self):
        """The rpc port is offset by this number, and only the supervisor's own variable carries it."""
        assert read_worker_in_pod_index({SUBPROCESS_INDEX_ENV_VAR: "3"}) == 3

    def test_an_unsupervised_process_is_the_only_worker_of_its_pod(self):
        """A worker launched without the supervisor must still bind the port the address book predicts."""
        assert read_worker_in_pod_index({}) == 0
