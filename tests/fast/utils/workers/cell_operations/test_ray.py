from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.context_lock import ContextLock
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.ray import RayCellOperations


class _RecordingEngineProvider:
    def __init__(self) -> None:
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


@dataclass(frozen=True)
class _Fixture:
    provider: _RecordingEngineProvider
    controller: InferenceController
    worker_manager: _RecordingWorkerManagerHandle
    operations: RayCellOperations


def _make_fixture() -> _Fixture:
    provider = _RecordingEngineProvider()
    controller = InferenceController(SimpleNamespace(), engine_provider=provider, router_providers=[])
    worker_manager = _RecordingWorkerManagerHandle()
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


async def test_every_other_operation_goes_straight_through() -> None:
    """Only a suspend takes a rank out of a live collective, so nothing else pays for the lock."""
    fixture = _make_fixture()
    acquired, release = asyncio.Event(), asyncio.Event()
    holding = asyncio.create_task(_hold_lock(lock=fixture.controller.context_lock, acquired=acquired, release=release))
    await acquired.wait()

    await asyncio.wait_for(fixture.operations.cell_infos(pool_ids=["engine-0"]), timeout=5.0)
    await asyncio.wait_for(fixture.operations.resume(cell_id="engine-0-2"), timeout=5.0)
    await asyncio.wait_for(
        fixture.operations.inject_fault(cell_id="engine-0-2", mode=FailureMode.SIGKILL, sub_index=0),
        timeout=5.0,
    )

    assert [name for name, _, _ in fixture.worker_manager.calls] == ["get_cell_infos", "start_cells", "inject_fault"]
    assert fixture.provider.stopped == []

    release.set()
    await holding
