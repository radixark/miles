from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import pytest

import miles.utils.workers.worker_provider.ray as ray_worker_provider_mod
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
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


async def _resolved(value: Any) -> Any:
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
class _FakeStopCellsMethod:
    calls: list[tuple[list[str], dict[str, str] | None]] = field(default_factory=list)

    def remote(
        self,
        cell_ids: list[str],
        *,
        expected_workers_hashes: dict[str, str] | None = None,
    ) -> Any:
        self.calls.append((cell_ids, expected_workers_hashes))
        return _resolved(None)


@dataclass
class _StoppingManagerHandle:
    stop_cells: _FakeStopCellsMethod


class TestRayWorkerProviderStopCells:
    async def test_the_expected_generation_reaches_the_manager(self):
        """The membership lock needs the snapshot hash to exclude a replacement atomically."""
        handle = _StoppingManagerHandle(stop_cells=_FakeStopCellsMethod())
        provider = RayWorkerProvider(worker_manager_handle=handle)

        await provider.stop_cells(
            cell_ids=["engine-0"],
            expected_workers_hashes={"engine-0": "hash-old"},
        )

        assert handle.stop_cells.calls == [(["engine-0"], {"engine-0": "hash-old"})]


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


def _worker_infos(cell_id: str, *, count: int) -> list[WorkerInfo]:
    return [
        WorkerInfo(
            name=f"{cell_id}-{worker_index}",
            generation=1,
            self_addrs={"primary": HostAndPort(host="10.0.0.7", port=15000 + worker_index)},
            gpu_ids=[worker_index],
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


class _RpcDemoWorker:
    def report(self) -> str:
        return "ok"


_RPC_DEMO_WORKER_PATH = f"{__name__}._RpcDemoWorker"


@dataclass
class _ServingWorkerInfosMethod:
    answers: list[list[WorkerInfo]]
    calls: list[str] = field(default_factory=list)

    def remote(self, cell_id: str) -> Any:
        self.calls.append(cell_id)
        return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]


@dataclass
class _ServingManagerHandle:
    get_worker_infos: _ServingWorkerInfosMethod


def _served_worker_info(*, generation: int, port: int = 15000) -> WorkerInfo:
    return WorkerInfo(
        name="trainer-engine-actor-00000-00000",
        generation=generation,
        self_addrs={"rpc": HostAndPort(host="10.0.0.7", port=port)},
        gpu_ids=[],
        worker_class=_RPC_DEMO_WORKER_PATH,
    )


class TestRayWorkerProviderRpcHandles:
    def test_a_served_worker_is_called_over_its_own_server(self, monkeypatch: pytest.MonkeyPatch):
        """Under rpc the launcher answers with the class, and the caller is what builds the client."""
        handle = _ServingManagerHandle(
            get_worker_infos=_ServingWorkerInfosMethod(answers=[[_served_worker_info(generation=0)]])
        )
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["trainer-engine-actor"])
        monkeypatch.setattr(ray_worker_provider_mod.ray, "get", lambda refs: refs)

        built = provider.get_handle("trainer-engine-actor-00000-00000")

        assert isinstance(built, RpcWorkerHandle)
        assert built._transport._server_url == "http://10.0.0.7:15000"

    def test_the_worker_of_the_named_cell_is_asked_for(self, monkeypatch: pytest.MonkeyPatch):
        """A lookup that derived another cell would hand back a handle to somebody else's worker."""
        handle = _ServingManagerHandle(
            get_worker_infos=_ServingWorkerInfosMethod(answers=[[_served_worker_info(generation=0)]])
        )
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["trainer-engine-actor"])
        monkeypatch.setattr(ray_worker_provider_mod.ray, "get", lambda refs: refs)

        provider.get_handle("trainer-engine-actor-00000-00000")

        assert handle.get_worker_infos.calls == ["trainer-engine-actor-00000"]


@dataclass
class _ActorHandleMethod:
    generation_of_worker: int
    requested: list[tuple[str, int]] = field(default_factory=list)

    def remote(self, worker_name: str, *, expected_generation: int) -> Any:
        self.requested.append((worker_name, expected_generation))
        assert expected_generation == self.generation_of_worker, f"generation {self.generation_of_worker}"
        return f"actor-of-{worker_name}"


@dataclass
class _RayCommManagerHandle:
    get_worker_infos: _ServingWorkerInfosMethod
    get_actor_handle: _ActorHandleMethod


def _actor_worker_info(*, generation: int, name: str = "trainer-engine-actor-00000-00000") -> WorkerInfo:
    return WorkerInfo(
        name=name,
        generation=generation,
        self_addrs={"master": HostAndPort(host="10.0.0.7", port=20000)},
        gpu_ids=[],
        worker_class=None,
    )


class TestRayWorkerProviderRayHandlesAreOfTheGenerationDescribed:
    def test_a_handle_of_the_described_generation_is_built(self, monkeypatch: pytest.MonkeyPatch):
        """The ordinary path: the cell did not move between the two calls the resolution takes."""
        handle = _RayCommManagerHandle(
            get_worker_infos=_ServingWorkerInfosMethod(answers=[[_actor_worker_info(generation=3)]]),
            get_actor_handle=_ActorHandleMethod(generation_of_worker=3),
        )
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["trainer-engine-actor"])
        monkeypatch.setattr(ray_worker_provider_mod.ray, "get", lambda ref: ref)

        provider.get_handle("trainer-engine-actor-00000-00000")

        assert handle.get_actor_handle.requested == [("trainer-engine-actor-00000-00000", 3)]

    def test_a_cell_restarted_between_the_two_calls_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        """Pairing an old cell's addresses with a new cell's actors sends new ranks to a dead rendezvous."""
        handle = _RayCommManagerHandle(
            get_worker_infos=_ServingWorkerInfosMethod(answers=[[_actor_worker_info(generation=3)]]),
            get_actor_handle=_ActorHandleMethod(generation_of_worker=4),
        )
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["trainer-engine-actor"])
        monkeypatch.setattr(ray_worker_provider_mod.ray, "get", lambda ref: ref)

        with pytest.raises(AssertionError):
            provider.get_handle("trainer-engine-actor-00000-00000")


@dataclass
class _PendingActorHandle:
    worker_name: str
    expected_generation: int


@dataclass
class _PendingActorHandleMethod:
    requested: list[tuple[str, int]] = field(default_factory=list)

    def remote(self, worker_name: str, *, expected_generation: int) -> _PendingActorHandle:
        self.requested.append((worker_name, expected_generation))
        return _PendingActorHandle(worker_name=worker_name, expected_generation=expected_generation)


@dataclass
class _PendingManagerHandle:
    get_actor_handle: _PendingActorHandleMethod


@dataclass
class _RecordingRayGet:
    generation_of_workers: dict[str, int]
    batches: list[list[_PendingActorHandle]] = field(default_factory=list)

    def __call__(self, refs: list[_PendingActorHandle]) -> list[str]:
        self.batches.append(refs)
        return [self._resolve(ref) for ref in refs]

    def _resolve(self, ref: _PendingActorHandle) -> str:
        generation = self.generation_of_workers[ref.worker_name]
        assert (
            generation == ref.expected_generation
        ), f"{ref.worker_name} is now generation {generation}, not the {ref.expected_generation} it was described as"
        return f"actor-of-{ref.worker_name}"


class _BatchingProbe(NamedTuple):
    provider: RayWorkerProvider
    manager_handle: _PendingManagerHandle
    ray_get: _RecordingRayGet


def _build_batching_provider(
    *, monkeypatch: pytest.MonkeyPatch, generation_of_workers: dict[str, int]
) -> _BatchingProbe:
    handle = _PendingManagerHandle(get_actor_handle=_PendingActorHandleMethod())
    provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["trainer-engine-actor"])
    ray_get = _RecordingRayGet(generation_of_workers=generation_of_workers)
    monkeypatch.setattr(ray_worker_provider_mod.ray, "get", ray_get)
    return _BatchingProbe(provider=provider, manager_handle=handle, ray_get=ray_get)


class TestRayWorkerProviderHandlesAreFetchedInOneBatch:
    def test_many_workers_cost_a_single_round_trip(self, monkeypatch: pytest.MonkeyPatch):
        """Resolving per worker blocks the event loop for one round trip each, so all of them go out together."""
        names = [f"trainer-engine-actor-00000-{index:05d}" for index in range(4)]
        probe = _build_batching_provider(monkeypatch=monkeypatch, generation_of_workers={name: 3 for name in names})

        probe.provider.get_handles_of_worker_infos([_actor_worker_info(name=name, generation=3) for name in names])

        assert len(probe.ray_get.batches) == 1
        assert probe.manager_handle.get_actor_handle.requested == [(name, 3) for name in names]

    def test_every_worker_is_paired_with_its_own_handle(self, monkeypatch: pytest.MonkeyPatch):
        """A batch that mixes up the order would hand each caller somebody else's actor."""
        names = [f"trainer-engine-actor-00000-{index:05d}" for index in range(3)]
        probe = _build_batching_provider(monkeypatch=monkeypatch, generation_of_workers={name: 3 for name in names})

        handles = probe.provider.get_handles_of_worker_infos(
            [_actor_worker_info(name=name, generation=3) for name in names]
        )

        assert {name: handle._actor_handle for name, handle in handles.items()} == {
            name: f"actor-of-{name}" for name in names
        }

    def test_a_worker_of_another_generation_is_still_refused(self, monkeypatch: pytest.MonkeyPatch):
        """Batching must not swallow the generation check that keeps stale actors out of a new cell."""
        names = ["trainer-engine-actor-00000-00000", "trainer-engine-actor-00000-00001"]
        probe = _build_batching_provider(monkeypatch=monkeypatch, generation_of_workers={names[0]: 3, names[1]: 4})

        with pytest.raises(AssertionError, match=names[1]):
            probe.provider.get_handles_of_worker_infos([_actor_worker_info(name=name, generation=3) for name in names])

    def test_asking_for_nothing_answers_nothing(self, monkeypatch: pytest.MonkeyPatch):
        """An empty batch must answer an empty mapping instead of failing on a round trip with nothing to ask."""
        probe = _build_batching_provider(monkeypatch=monkeypatch, generation_of_workers={})

        assert probe.provider.get_handles_of_worker_infos([]) == {}
        assert probe.manager_handle.get_actor_handle.requested == []

    def test_served_workers_keep_their_locally_built_handles(self, monkeypatch: pytest.MonkeyPatch):
        """An rpc worker needs no round trip, and must not be dropped from a batch that also holds actors."""
        actor_name = "trainer-engine-actor-00000-00001"
        probe = _build_batching_provider(monkeypatch=monkeypatch, generation_of_workers={actor_name: 3})

        handles = probe.provider.get_handles_of_worker_infos(
            [_served_worker_info(generation=3), _actor_worker_info(name=actor_name, generation=3)]
        )

        assert isinstance(handles["trainer-engine-actor-00000-00000"], RpcWorkerHandle)
        assert probe.manager_handle.get_actor_handle.requested == [(actor_name, 3)]


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
