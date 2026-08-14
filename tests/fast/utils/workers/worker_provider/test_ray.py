from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

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


class TestRayWorkerProviderCreate:
    async def test_looks_addresses_up_through_the_named_manager_actor(self, monkeypatch: pytest.MonkeyPatch):
        """The provider finds the manager by its well-known actor name and asks it for the address."""
        import miles.utils.workers.ray_worker_manager as ray_worker_manager_mod

        handle = _make_handle({"primary": HostAndPort(host="10.0.0.7", port=15000)})
        looked_up: list[str] = []

        class _FakeRay:
            @staticmethod
            def get_actor(name: str) -> _FakeManagerHandle:
                looked_up.append(name)
                return handle

        monkeypatch.setattr(ray_worker_manager_mod, "ray", _FakeRay)

        provider = RayWorkerProvider.create()
        addr = (await provider.get_addrs(worker_name="router-0-0"))["primary"]

        assert looked_up == ["ray_worker_manager"]
        assert handle.get_worker_addrs.requested_names == ["router-0-0"]
        assert addr == HostAndPort(host="10.0.0.7", port=15000)


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


class TestRayWorkerProviderWatchCellsStop:
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
