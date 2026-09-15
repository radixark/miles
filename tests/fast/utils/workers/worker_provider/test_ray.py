from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

import miles.utils.workers.worker_provider.ray as ray_worker_provider_mod
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort


@dataclass
class _FakeRemoteMethod:
    answers: list[dict[str, HostAndPort]]
    requested_names: list[str] = field(default_factory=list)

    def remote(self, worker_name: str) -> Any:
        self.requested_names.append(worker_name)
        return _resolved(self.answers[len(self.requested_names) - 1])


@dataclass
class _FakeManagerHandle:
    get_worker_addrs: _FakeRemoteMethod


async def _resolved(value: dict[str, HostAndPort]) -> dict[str, HostAndPort]:
    return value


def _make_handle(*answers: dict[str, HostAndPort]) -> _FakeManagerHandle:
    return _FakeManagerHandle(get_worker_addrs=_FakeRemoteMethod(answers=list(answers)))


class TestRayWorkerProviderAddressLookup:
    async def test_every_lookup_asks_the_manager_again(self):
        """Addresses are never cached, so a relaunched worker is not answered with a stale endpoint."""
        handle = _make_handle(
            {"primary": HostAndPort(host="10.0.0.7", port=15000)},
            {"primary": HostAndPort(host="10.0.0.7", port=15001)},
        )
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])

        first = (await provider.get_addrs(worker_name="router-0-0"))["primary"]
        second = (await provider.get_addrs(worker_name="router-0-0"))["primary"]

        assert (first.port, second.port) == (15000, 15001)
        assert handle.get_worker_addrs.requested_names == ["router-0-0", "router-0-0"]


class TestRayWorkerProviderGetAddrs:
    async def test_returns_every_named_port_of_the_worker(self):
        """Consumers that need more than the primary endpoint get the worker's whole address map."""
        addrs = {
            "primary": HostAndPort(host="10.0.0.7", port=15000),
            "disaggregation_bootstrap": HostAndPort(host="10.0.0.7", port=15001),
        }
        handle = _make_handle(addrs)
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])

        assert await provider.get_addrs(worker_name="engine-0-0") == addrs


@dataclass
class _FakeObjectRef:
    infos: list[WorkerInfo]


@dataclass
class _FakeWorkerInfosMethod:
    infos_by_cell_id: dict[str, list[WorkerInfo]]
    requested_cell_ids: list[str] = field(default_factory=list)

    def remote(self, cell_id: str) -> _FakeObjectRef:
        self.requested_cell_ids.append(cell_id)
        return _FakeObjectRef(infos=self.infos_by_cell_id[cell_id])


@dataclass
class _WorkerInfosManagerHandle:
    get_worker_infos: _FakeWorkerInfosMethod


class _FakeRayModule:
    @staticmethod
    def get(refs: list[_FakeObjectRef]) -> list[list[WorkerInfo]]:
        return [ref.infos for ref in refs]


class _FakeWorkerHandle(BaseWorkerHandle):
    async def wait_ready(self, *, timeout: float) -> None:
        return None

    async def wait_dead(self, *, timeout: float) -> None:
        return None


def _worker_infos(cell_id: str, *, count: int) -> list[WorkerInfo]:
    return [
        WorkerInfo(
            name=f"{cell_id}-{worker_index}",
            generation=1,
            self_addrs={"primary": HostAndPort(host="10.0.0.7", port=15000 + worker_index)},
            gpu_ids=[worker_index],
            handle=_FakeWorkerHandle(),
        )
        for worker_index in range(count)
    ]


class TestRayWorkerProviderGetWorkerInfos:
    def test_multiple_cell_ids_return_worker_info_groups_in_request_order(self, monkeypatch: pytest.MonkeyPatch):
        """Every requested cell gets its own worker group, positioned as the caller asked for it."""
        infos_by_cell_id = {
            "cell-a": _worker_infos("cell-a", count=1),
            "cell-b": _worker_infos("cell-b", count=2),
            "cell-c": _worker_infos("cell-c", count=3),
        }
        handle = _WorkerInfosManagerHandle(get_worker_infos=_FakeWorkerInfosMethod(infos_by_cell_id=infos_by_cell_id))
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])
        monkeypatch.setattr(ray_worker_provider_mod, "ray", _FakeRayModule)

        groups = provider.get_worker_infos(cell_ids=["cell-c", "cell-a", "cell-b"])

        assert handle.get_worker_infos.requested_cell_ids == ["cell-c", "cell-a", "cell-b"]
        assert groups == [infos_by_cell_id["cell-c"], infos_by_cell_id["cell-a"], infos_by_cell_id["cell-b"]]


@dataclass
class _FakeCellInfosMethod:
    answers: list[Any]
    calls: list[list[str]] = field(default_factory=list)

    def remote(self, *, pool_ids: list[str]) -> Any:
        self.calls.append(list(pool_ids))
        answer = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        if isinstance(answer, Exception):
            return _raised(answer)
        return _resolved(answer)


@dataclass
class _WatchingManagerHandle:
    get_cell_infos: _FakeCellInfosMethod


async def _raised(error: Exception) -> Any:
    raise error


def _make_watching_handle(*answers: Any) -> _WatchingManagerHandle:
    return _WatchingManagerHandle(get_cell_infos=_FakeCellInfosMethod(answers=list(answers)))


def _cell_info(cell_id: str, *, alive: bool = True, workers_hash: str = "hash-0") -> CellInfo:
    return CellInfo(
        cell_id=cell_id,
        pool_id="inference-engine-0-0",
        alive=alive,
        worker_names=[f"{cell_id}-0"],
        workers_hash=workers_hash,
        meta={},
    )


async def _wait_until(predicate, *, timeout_seconds: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the watch loop"
        await asyncio.sleep(0.001)


class _RecordingReconciler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, CellInfo | None]] = []

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        self.calls.append((cell_id, info))


class _FailingOnceReconciler(_RecordingReconciler):
    def __init__(self, *, failing_cell_id: str) -> None:
        super().__init__()
        self._failing_cell_id: str | None = failing_cell_id

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        if cell_id == self._failing_cell_id:
            self._failing_cell_id = None
            raise RuntimeError("reconcile failed")
        await super().__call__(cell_id, info)


class _FailingForCellReconciler(_RecordingReconciler):
    def __init__(self, *, failing_cell_id: str) -> None:
        super().__init__()
        self._failing_cell_id = failing_cell_id

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        await super().__call__(cell_id, info)
        if cell_id == self._failing_cell_id:
            raise RuntimeError(f"reconcile rejected {cell_id}")


class _AlwaysFailingReconciler(_RecordingReconciler):
    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        await super().__call__(cell_id, info)
        raise RuntimeError("reconcile rejected the cell")


class _BlockingReconciler(_RecordingReconciler):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.cancelled = False

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        await super().__call__(cell_id, info)
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _StuckOnCancelReconciler(_RecordingReconciler):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        await super().__call__(cell_id, info)
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self.release.wait()
            raise


class TestRayWorkerProviderWatchCellsInitialSync:
    async def test_every_initial_cell_is_reconciled_before_the_watch_is_established(self):
        """Callers may assume the pool is fully observed once watch_cells returns."""
        handle = _make_watching_handle({"cell-a": _cell_info("cell-a"), "cell-b": _cell_info("cell-b")})
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])
        reconciler = _RecordingReconciler()

        stop = await provider.watch_cells(reconciler)
        try:
            assert [cell_id for cell_id, _ in reconciler.calls] == ["cell-a", "cell-b"]
        finally:
            await stop()

    async def test_a_failing_initial_sync_propagates_instead_of_starting_the_loop(self):
        """A pool we never managed to read must not look like an empty pool."""
        handle = _make_watching_handle(RuntimeError("manager unreachable"))
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])

        with pytest.raises(RuntimeError, match="manager unreachable"):
            await provider.watch_cells(_RecordingReconciler())

    async def test_an_initial_reconcile_failure_prevents_watch_establishment(self):
        """A caller that never learned about the initial cells must not be left with a live watch."""
        handle = _make_watching_handle({"cell-a": _cell_info("cell-a")})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )

        with pytest.raises(RuntimeError, match="reconcile rejected the cell"):
            await provider.watch_cells(_AlwaysFailingReconciler())

        await asyncio.sleep(0.02)

        assert handle.get_cell_infos.calls == [["inference-engine-0-0"]]

    async def test_watching_without_pool_ids_fails_before_contacting_manager(self):
        """A provider built for address lookups only must refuse to watch rather than watch nothing."""
        handle = _make_watching_handle({})
        provider = RayWorkerProvider(worker_manager_handle=handle)

        with pytest.raises(AssertionError, match="without the pool_ids"):
            await provider.watch_cells(_RecordingReconciler())

        assert handle.get_cell_infos.calls == []

    async def test_only_the_requested_pools_are_asked_for(self):
        """The controller must not observe cells belonging to someone else."""
        handle = _make_watching_handle({})
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])

        stop = await provider.watch_cells(_RecordingReconciler())
        try:
            assert handle.get_cell_infos.calls == [["inference-engine-0-0"]]
        finally:
            await stop()


class TestRayWorkerProviderWatchCellsPolling:
    async def test_an_unchanged_cell_is_not_reconciled_again(self):
        """Re-reconciling every poll would restart cells every interval."""
        info = _cell_info("cell-a")
        handle = _make_watching_handle({"cell-a": info}, {"cell-a": info})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _RecordingReconciler()

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: len(handle.get_cell_infos.calls) >= 3)
            assert reconciler.calls == [("cell-a", info)]
        finally:
            await stop()

    async def test_a_stopped_cell_is_reported_as_gone_exactly_once(self):
        """A suspended cell must look like a disappeared cell, and must not be re-reported."""
        alive = _cell_info("cell-a")
        handle = _make_watching_handle({"cell-a": alive}, {"cell-a": _cell_info("cell-a", alive=False)})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _RecordingReconciler()

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: len(handle.get_cell_infos.calls) >= 4)
            assert reconciler.calls == [("cell-a", alive), ("cell-a", None)]
        finally:
            await stop()

    async def test_a_relaunched_cell_is_reconciled_again_because_its_workers_changed(self):
        """A replacement cell keeps its id, so only the workers hash can reveal it."""
        first = _cell_info("cell-a", workers_hash="hash-0")
        second = _cell_info("cell-a", workers_hash="hash-1")
        handle = _make_watching_handle({"cell-a": first}, {"cell-a": second})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _RecordingReconciler()

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: len(reconciler.calls) >= 2)
            assert reconciler.calls[:2] == [("cell-a", first), ("cell-a", second)]
        finally:
            await stop()

    async def test_a_failing_poll_is_retried_instead_of_killing_the_watch(self):
        """One unreachable manager call must not silently end pool observation."""
        info = _cell_info("cell-a")
        handle = _make_watching_handle({}, RuntimeError("transient"), {"cell-a": info})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _RecordingReconciler()

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: reconciler.calls == [("cell-a", info)])
        finally:
            await stop()

    async def test_a_cell_added_by_a_partially_failed_poll_can_still_disappear(self):
        """A reconcile that raises mid-poll must not lose the bookkeeping of the cells already delivered."""
        info_a = _cell_info("cell-a")
        info_b = _cell_info("cell-b")
        handle = _make_watching_handle({}, {"cell-a": info_a, "cell-b": info_b}, {"cell-b": info_b})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _FailingOnceReconciler(failing_cell_id="cell-b")

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: ("cell-a", None) in reconciler.calls)
            assert ("cell-a", info_a) in reconciler.calls
        finally:
            await stop()

    async def test_a_failed_reconcile_is_retried_before_the_cell_is_marked_seen(self):
        """A cell whose reconcile raised must be delivered again instead of being silently forgotten."""
        info = _cell_info("cell-a")
        handle = _make_watching_handle({}, {"cell-a": info})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _FailingOnceReconciler(failing_cell_id="cell-a")

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: reconciler.calls == [("cell-a", info)])
        finally:
            await stop()


class TestRayWorkerProviderWatchCellsStop:
    async def test_stopping_cancels_an_in_flight_reconcile(self):
        """Stopping must not hang waiting for a reconcile that is still running."""
        handle = _make_watching_handle({}, {"cell-a": _cell_info("cell-a")})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _BlockingReconciler()

        stop = await provider.watch_cells(reconciler)
        await asyncio.wait_for(reconciler.entered.wait(), timeout=2.0)
        await asyncio.wait_for(stop(), timeout=2.0)

        assert reconciler.cancelled

    async def test_stopping_ends_the_polling(self):
        """The returned stop function must actually stop the loop, not just detach from it."""
        handle = _make_watching_handle({})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )

        stop = await provider.watch_cells(_RecordingReconciler())
        await _wait_until(lambda: len(handle.get_cell_infos.calls) >= 2)
        await stop()
        settled = len(handle.get_cell_infos.calls)

        await asyncio.sleep(0.02)

        assert len(handle.get_cell_infos.calls) == settled

    async def test_stopping_does_not_swallow_the_callers_cancellation(self):
        """A stop that eats its caller's cancellation lets a timed-out shutdown run on past its teardown."""
        handle = _make_watching_handle({}, {"cell-a": _cell_info("cell-a")})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _StuckOnCancelReconciler()
        returned: list[str] = []

        stop = await provider.watch_cells(reconciler)
        await asyncio.wait_for(reconciler.entered.wait(), timeout=2.0)

        async def _stopper() -> None:
            await stop()
            returned.append("returned")

        stopper = asyncio.create_task(_stopper())
        await asyncio.sleep(0)
        stopper.cancel()
        await asyncio.gather(stopper, return_exceptions=True)
        reconciler.release.set()
        await stop()

        assert stopper.cancelled() and returned == []


class TestRayWorkerProviderPollIsolation:
    async def test_one_cells_failing_reconcile_does_not_block_the_others(self):
        """A cell that keeps failing to reconcile must not leave the rest of the pool unobserved."""
        handle = _make_watching_handle({"cell-a": _cell_info("cell-a"), "cell-b": _cell_info("cell-b")})
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])
        reconciler = _FailingForCellReconciler(failing_cell_id="cell-a")

        with pytest.raises(RuntimeError, match="reconcile rejected cell-a"):
            await provider.watch_cells(reconciler)

        assert [cell_id for cell_id, _ in reconciler.calls] == ["cell-a", "cell-b"]

    async def test_a_failed_cell_is_retried_while_the_others_are_not(self):
        """Only the cell whose reconcile failed stays unobserved for the next poll."""
        info_a = _cell_info("cell-a")
        info_b = _cell_info("cell-b")
        handle = _make_watching_handle({}, {"cell-a": info_a, "cell-b": info_b})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _FailingOnceReconciler(failing_cell_id="cell-a")

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: ("cell-a", info_a) in reconciler.calls)
        finally:
            await stop()

        assert reconciler.calls == [("cell-b", info_b), ("cell-a", info_a)]
