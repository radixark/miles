from __future__ import annotations

import asyncio

import pytest
import ray
from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.fake_ray import EVENT_KILL, READINESS_METHOD, FakeRayCluster

from miles.utils.workers import ray_worker_manager
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec


class DemoWorker:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


_WORKER_CLASS_PATH = f"{DemoWorker.__module__}.{DemoWorker.__qualname__}"


def _make_spec(name: str, *, num_cells: int = 1, num_workers_per_cell: int = 1) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=name,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master", allow_dynamic=True)],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=num_cells, num_workers_per_cell=num_workers_per_cell, num_gpus_per_worker=0
        ),
        worker_class=_WORKER_CLASS_PATH,
        ctor_kwargs=lambda _ctx: {},
    )


async def _launch(specs: list[ServeWorkerSpec], *, comm_backend: WorkerCommBackend) -> RayWorkerManager:
    manager = RayWorkerManager()
    await manager.init(worker_manager_args(), specs, {}, comm_backend=comm_backend)
    return manager


async def _scan_all_live_cells(manager: RayWorkerManager) -> None:
    for cell in manager._all_cells():
        if cell.alive:
            await cell._scan_liveness_once()


def _kill_worker_process(cluster: FakeRayCluster, *, handle_index: int) -> None:
    cluster.handles[handle_index].failing_methods[READINESS_METHOD] = ray.exceptions.RayActorError()


@pytest.fixture
def instant_scans(monkeypatch) -> None:
    monkeypatch.setattr(ray_worker_manager, "_LIVENESS_SCAN_INTERVAL_SECONDS", 0.0)


class TestScanLivenessOnce:
    async def test_a_cell_whose_workers_all_answer_stays_alive(self, fake_ray_cluster: FakeRayCluster):
        """The scan must not tear down a healthy cell, or every run would restart itself forever."""
        manager = await _launch([_make_spec("engine", num_cells=2)], comm_backend=WorkerCommBackend.RPC)

        await _scan_all_live_cells(manager)

        assert all(info.alive for info in manager.get_cell_infos(pool_ids=["engine"]).values())

    async def test_a_cell_that_lost_a_worker_stops_being_alive(self, fake_ray_cluster: FakeRayCluster):
        """A worker that exits on its own must reach the membership, or nobody ever replaces it."""
        manager = await _launch([_make_spec("engine", num_cells=2)], comm_backend=WorkerCommBackend.RPC)
        _kill_worker_process(fake_ray_cluster, handle_index=0)

        await _scan_all_live_cells(manager)

        infos = manager.get_cell_infos(pool_ids=["engine"])
        assert not infos["engine-00000"].alive
        assert infos["engine-00001"].alive

    async def test_the_whole_cell_goes_when_one_of_its_workers_dies(self, fake_ray_cluster: FakeRayCluster):
        """A cell is the unit of recovery, so a surviving sibling of a dead rank must be reclaimed too."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)], comm_backend=WorkerCommBackend.RPC)
        _kill_worker_process(fake_ray_cluster, handle_index=1)

        await _scan_all_live_cells(manager)

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive
        assert [handle.killed for handle in fake_ray_cluster.handles] == [True, True]

    async def test_a_dropped_cell_can_be_started_again(self, fake_ray_cluster: FakeRayCluster):
        """Reporting the death is only useful if the cell is then restartable without a stop first."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        _kill_worker_process(fake_ray_cluster, handle_index=0)
        await _scan_all_live_cells(manager)

        await manager.start_cells(["engine-00000"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive
        assert len(fake_ray_cluster.handles) == 2

    async def test_a_restarted_cell_reports_a_new_workers_hash(self, fake_ray_cluster: FakeRayCluster):
        """The consumer rebuilds its handles off the hash, so a self-death must move it as a stop does."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        before = manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].workers_hash
        _kill_worker_process(fake_ray_cluster, handle_index=0)

        await _scan_all_live_cells(manager)
        await manager.start_cells(["engine-00000"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].workers_hash != before

    async def test_a_ray_wire_cell_is_scanned_the_same_way(self, fake_ray_cluster: FakeRayCluster):
        """Liveness is a property of the actor process, not of the wire its methods travel on."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RAY)
        _kill_worker_process(fake_ray_cluster, handle_index=0)

        await _scan_all_live_cells(manager)

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive

    async def test_a_stopped_cell_probes_nothing(self, fake_ray_cluster: FakeRayCluster):
        """A scan racing a suspend finds no actors to probe, and must answer that instead of raising."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        cell = manager._find_cell("engine-00000")
        await manager.stop_cells(["engine-00000"])
        fake_ray_cluster.calls.clear()

        await cell._scan_liveness_once()

        assert fake_ray_cluster.calls_of(READINESS_METHOD) == []


class TestScanLivenessRacesWithMembershipChanges:
    async def test_a_cell_stopped_while_it_is_probed_is_not_stopped_twice(self, fake_ray_cluster: FakeRayCluster):
        """A suspend landing mid-probe already killed the actors the scan was about to declare dead."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        cell = manager._find_cell("engine-00000")
        _kill_worker_process(fake_ray_cluster, handle_index=0)
        fake_ray_cluster.handles[0].hanging_methods[READINESS_METHOD] = 0.2

        scan = asyncio.create_task(cell._scan_liveness_once())
        await asyncio.sleep(0.05)
        await manager.stop_cells(["engine-00000"])
        await scan

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive
        assert fake_ray_cluster.events.count(EVENT_KILL) == 1

    async def test_a_cell_restarted_while_being_probed_survives(self, fake_ray_cluster: FakeRayCluster):
        """The dead workers the scan saw belong to the old generation, so the new one must not pay for them."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        cell = manager._find_cell("engine-00000")
        _kill_worker_process(fake_ray_cluster, handle_index=0)
        fake_ray_cluster.handles[0].hanging_methods[READINESS_METHOD] = 0.2

        scan = asyncio.create_task(cell._scan_liveness_once())
        await asyncio.sleep(0.05)
        await manager.stop_cells(["engine-00000"])
        await manager.start_cells(["engine-00000"])
        await scan

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive


class TestScanLivenessOnlyTrustsAProvenDeath:
    async def test_a_worker_that_does_not_answer_in_time_is_kept(self, fake_ray_cluster: FakeRayCluster):
        """A busy worker must not be declared dead, or a slow train step would kill its own cell."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        fake_ray_cluster.handles[0].hanging_methods[READINESS_METHOD] = 3600.0

        await _scan_all_live_cells(manager)

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive

    async def test_an_application_error_from_the_probe_is_treated_as_death(self, fake_ray_cluster: FakeRayCluster):
        """A worker answering its readiness probe with a task error is as unusable as a missing one."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        fake_ray_cluster.handles[0].failing_methods[READINESS_METHOD] = ray.exceptions.RayTaskError.__new__(
            ray.exceptions.RayTaskError
        )

        await _scan_all_live_cells(manager)

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive


class TestScanLivenessLoop:
    async def test_a_launched_cell_scans_itself(self, fake_ray_cluster: FakeRayCluster, instant_scans: None):
        """The scan is what makes a death nobody reported reach the membership, so it must run unasked."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        _kill_worker_process(fake_ray_cluster, handle_index=0)

        await _wait_until(lambda: not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive)

    async def test_each_cell_scans_on_its_own(self, fake_ray_cluster: FakeRayCluster, instant_scans: None):
        """One worker hanging its probe must not delay how fast another cell's death is noticed."""
        manager = await _launch([_make_spec("engine", num_cells=2)], comm_backend=WorkerCommBackend.RPC)
        fake_ray_cluster.handles[0].hanging_methods[READINESS_METHOD] = 3600.0
        _kill_worker_process(fake_ray_cluster, handle_index=1)

        await _wait_until(lambda: not manager.get_cell_infos(pool_ids=["engine"])["engine-00001"].alive)

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive

    async def test_a_stopped_cell_stops_scanning(self, fake_ray_cluster: FakeRayCluster, instant_scans: None):
        """A suspended cell that kept scanning would keep a task per suspend for the rest of the run."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        cell = manager._find_cell("engine-00000")

        await manager.stop_cells(["engine-00000"])

        await _wait_until(lambda: cell.liveness_scan_task.done())

    async def test_a_restarted_cell_leaves_only_its_new_scan_running(
        self, fake_ray_cluster: FakeRayCluster, instant_scans: None
    ):
        """A restart that stacked one more scan per generation would probe the same actors N times."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        first_scan = manager._find_cell("engine-00000").liveness_scan_task

        await manager.stop_cells(["engine-00000"])
        await manager.start_cells(["engine-00000"])

        await _wait_until(lambda: first_scan.done())
        assert not manager._find_cell("engine-00000").liveness_scan_task.done()

    async def test_a_failing_scan_does_not_end_the_loop(self, fake_ray_cluster: FakeRayCluster, instant_scans: None):
        """One bad scan must not silently leave the cell without any liveness reporting at all."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        cell = manager._find_cell("engine-00000")
        scans: list[int] = []

        async def scan_once() -> None:
            scans.append(len(scans))
            if len(scans) == 1:
                raise RuntimeError("scan failed")

        cell._scan_liveness_once = scan_once

        await _wait_until(lambda: len(scans) >= 3)

        assert not cell.liveness_scan_task.done()


class _ManagerHandleShim:
    """The provider talks to the manager as a ray actor; here it is the object itself."""

    def __init__(self, manager: RayWorkerManager) -> None:
        self.get_cell_infos = _ShimMethod(manager.get_cell_infos)


class _ShimMethod:
    def __init__(self, fn) -> None:
        self._fn = fn

    def remote(self, **kwargs):
        return _resolved(self._fn(**kwargs))


async def _resolved(value):
    return value


class _RecordingReconciler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []

    async def __call__(self, cell_id: str, info) -> None:
        self.calls.append((cell_id, info))


class TestTheDeathReachesTheProvider:
    async def test_a_worker_that_died_alone_is_reconciled_away(self, fake_ray_cluster: FakeRayCluster):
        """Noticing the death is only worth anything if the watcher that rebuilds cells hears about it."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        provider = RayWorkerProvider(
            worker_manager_handle=_ManagerHandleShim(manager),
            pool_ids=["engine"],
            poll_interval_seconds=0.001,
        )
        reconciler = _RecordingReconciler()
        stop = await provider.watch_cells(reconciler)
        try:
            _kill_worker_process(fake_ray_cluster, handle_index=0)
            await _scan_all_live_cells(manager)

            await _wait_until(lambda: reconciler.calls[-1][1] is None)
        finally:
            await stop()

        assert [cell_id for cell_id, _ in reconciler.calls] == ["engine-00000", "engine-00000"]

    async def test_the_rebuilt_cell_is_reconciled_back_with_a_new_hash(self, fake_ray_cluster: FakeRayCluster):
        """A cell dropped by the scan must be startable again and look new, or nothing reconnects to it."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        provider = RayWorkerProvider(
            worker_manager_handle=_ManagerHandleShim(manager),
            pool_ids=["engine"],
            poll_interval_seconds=0.001,
        )
        reconciler = _RecordingReconciler()
        stop = await provider.watch_cells(reconciler)
        try:
            first_hash = reconciler.calls[0][1].workers_hash
            _kill_worker_process(fake_ray_cluster, handle_index=0)
            await _scan_all_live_cells(manager)
            await _wait_until(lambda: reconciler.calls[-1][1] is None)

            await manager.start_cells(["engine-00000"])

            await _wait_until(lambda: reconciler.calls[-1][1] is not None)
        finally:
            await stop()

        assert reconciler.calls[-1][1].workers_hash != first_hash


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "the condition never became true"
        await asyncio.sleep(0.01)
