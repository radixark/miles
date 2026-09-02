from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE
from miles.utils.workers.cell_operations.ray import RayCellOperations


class _RecordingEngineProvider:
    def __init__(self, *, worker_manager: _RecordingWorkerManagerHandle) -> None:
        self._worker_manager_handle = worker_manager
        self.stopped: list[str] = []

    async def stop_cells(self, *, cell_ids: list[str]) -> None:
        self.stopped.extend(cell_ids)


class _RecordingRemoteMethod:
    def __init__(self, *, name: str, calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> None:
        self._name = name
        self._calls = calls

    async def remote(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._calls.append((self._name, args, kwargs))
        return {}


class _RecordingWorkerManagerHandle:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.get_cell_infos = _RecordingRemoteMethod(name="get_cell_infos", calls=self.calls)
        self.start_cells = _RecordingRemoteMethod(name="start_cells", calls=self.calls)
        self.inject_fault = _RecordingRemoteMethod(name="inject_fault", calls=self.calls)


class _RecordingRolloutServer:
    def __init__(self, *, cell_ids: list[str]) -> None:
        self.server_cells = {cell_id: SimpleNamespace() for cell_id in cell_ids}
        self.health_checker_activeness = ActivenessTracker(active=True)
        self.model_name = "model"
        self.faulted_cells: list[str] = []

    def addressable_cell_ids(self) -> list[str]:
        return [cell_id for cell_id in self.server_cells if cell_id not in self.faulted_cells]

    def mark_cell_faulted(self, cell_id: str) -> None:
        self.faulted_cells.append(cell_id)


@dataclass(frozen=True)
class _Fixture:
    provider: _RecordingEngineProvider
    controller: InferenceController
    worker_manager: _RecordingWorkerManagerHandle
    operations: RayCellOperations


def _make_fixture() -> _Fixture:
    worker_manager = _RecordingWorkerManagerHandle()
    provider = _RecordingEngineProvider(worker_manager=worker_manager)
    controller = InferenceController(SimpleNamespace(), engine_provider=provider, router_providers=[])
    controller.servers = {"actor": _RecordingRolloutServer(cell_ids=["engine-0-2", "engine-0-3"])}
    return _Fixture(
        provider=provider,
        controller=controller,
        worker_manager=worker_manager,
        operations=RayCellOperations(
            worker_manager_handle=worker_manager, resolve_inference_controller=lambda: controller
        ),
    )


async def _hold_lock(*, lock: ContextLock, acquired: asyncio.Event, release: asyncio.Event) -> None:
    async with lock:
        acquired.set()
        await release.wait()


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_a_suspend_waits_for_the_controller_lock_instead_of_reaching_the_worker_manager() -> None:
    """A suspend arriving mid weight update must not reach the worker manager until the update ends."""
    fixture = _make_fixture()
    acquired, release = asyncio.Event(), asyncio.Event()
    holding = asyncio.create_task(_hold_lock(lock=fixture.controller.context_lock, acquired=acquired, release=release))
    await acquired.wait()

    suspending = asyncio.create_task(fixture.operations.suspend(cell_id="engine-0-2"))
    await _settle()
    assert not suspending.done()
    assert fixture.provider.stopped == []
    assert fixture.worker_manager.calls == []

    release.set()
    await holding
    await suspending
    assert fixture.provider.stopped == ["engine-0-2"]
    assert fixture.worker_manager.calls == []


async def test_inject_fault_waits_for_the_controller_lock() -> None:
    """A fault arriving mid weight update must wait before killing an engine worker."""
    fixture = _make_fixture()
    acquired, release = asyncio.Event(), asyncio.Event()
    holding = asyncio.create_task(_hold_lock(lock=fixture.controller.context_lock, acquired=acquired, release=release))
    await acquired.wait()

    injecting = asyncio.create_task(
        fixture.operations.inject_fault(
            cell_id="engine-0-2", cell_type=ROLLOUT_CELL_TYPE, mode=FailureMode.SIGKILL, sub_index=0
        )
    )
    await _settle()
    assert not injecting.done()
    assert fixture.worker_manager.calls == []

    release.set()
    await holding
    await injecting
    assert fixture.worker_manager.calls == [
        (
            "inject_fault",
            ("engine-0-2",),
            {"mode": "sigkill", "worker_in_cell_index": 0, "wait_until_applied": True},
        )
    ]


async def test_non_disruptive_operations_go_straight_through() -> None:
    """Cell reads and resumes do not wait for the weight-update lock."""
    fixture = _make_fixture()
    acquired, release = asyncio.Event(), asyncio.Event()
    holding = asyncio.create_task(_hold_lock(lock=fixture.controller.context_lock, acquired=acquired, release=release))
    await acquired.wait()

    await asyncio.wait_for(fixture.operations.cell_infos(pool_ids=["engine-0"]), timeout=5.0)
    await asyncio.wait_for(fixture.operations.resume(cell_id="engine-0-2"), timeout=5.0)

    assert [name for name, _, _ in fixture.worker_manager.calls] == ["get_cell_infos", "start_cells"]
    assert fixture.provider.stopped == []

    release.set()
    await holding


async def test_an_actor_fault_reaches_the_worker_manager_without_the_rollout_controller() -> None:
    """The rollout controller knows no actor cells, so routing one through it would only raise KeyError."""
    fixture = _make_fixture()
    acquired, release = asyncio.Event(), asyncio.Event()
    holding = asyncio.create_task(_hold_lock(lock=fixture.controller.context_lock, acquired=acquired, release=release))
    await acquired.wait()

    await asyncio.wait_for(
        fixture.operations.inject_fault(
            cell_id="trainer-engine-actor-00000", cell_type=ACTOR_CELL_TYPE, mode=FailureMode.SIGKILL, sub_index=1
        ),
        timeout=5.0,
    )

    assert fixture.worker_manager.calls == [
        ("inject_fault", ("trainer-engine-actor-00000",), {"mode": "sigkill", "worker_in_cell_index": 1})
    ]

    release.set()
    await holding


async def test_suspending_a_trainer_cell_does_not_need_a_rollout_server() -> None:
    """Every cell type is routed through the rollout controller for its lock; only rollout cells live in a server."""
    fixture = _make_fixture()

    await fixture.operations.suspend(cell_id="trainer-engine-actor-00000")

    assert fixture.provider.stopped == ["trainer-engine-actor-00000"]


async def test_suspending_a_rollout_cell_reconcile_already_dropped_still_stops_it() -> None:
    """Reconcile removes a killed cell before the heal loop suspends it, and the heal must survive that."""
    fixture = _make_fixture()
    del fixture.controller.servers["actor"].server_cells["engine-0-2"]

    await fixture.operations.suspend(cell_id="engine-0-2")

    assert fixture.provider.stopped == ["engine-0-2"]


async def test_suspending_a_tracked_rollout_cell_takes_it_out_of_the_fleet() -> None:
    """A suspension is a kill, so the next weight update must not snapshot the cell it tore down."""
    fixture = _make_fixture()

    await fixture.operations.suspend(cell_id="engine-0-2")

    assert fixture.controller.servers["actor"].faulted_cells == ["engine-0-2"]
