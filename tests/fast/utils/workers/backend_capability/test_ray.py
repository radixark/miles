from __future__ import annotations

from dataclasses import dataclass

import pytest

import miles.utils.workers.worker_provider.ray as ray_worker_provider_mod
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_spec import HostAndPort


class _FakeRemoteMethod:
    def __init__(self) -> None:
        self.suspended_cell_ids: list[str] = []

    async def remote(self, *, cell_id: str) -> None:
        self.suspended_cell_ids.append(cell_id)


@dataclass
class _FakeInferenceControllerActor:
    stop_cell_between_weight_updates: _FakeRemoteMethod


@dataclass
class _FakeGetWorkerInfosMethod:
    controller_info: WorkerInfo

    def remote(self, cell_id: str) -> list[WorkerInfo]:
        return [self.controller_info]


@dataclass
class _FakeGetActorHandleMethod:
    controller: _FakeInferenceControllerActor

    def remote(self, worker_name: str, *, expected_generation: int) -> _FakeInferenceControllerActor:
        return self.controller


@dataclass
class _FakeWorkerManagerHandle:
    get_worker_infos: _FakeGetWorkerInfosMethod
    get_actor_handle: _FakeGetActorHandleMethod


class TestRayBackendCapabilityCellOperations:
    async def test_suspend_reaches_the_inference_controller_served_by_the_worker_manager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Suspending through a capability reaches its registered inference controller."""
        controller = _FakeInferenceControllerActor(stop_cell_between_weight_updates=_FakeRemoteMethod())
        controller_info = WorkerInfo(
            name="inference-controller-00000-00000",
            generation=3,
            self_addrs={"primary": HostAndPort(host="10.0.0.7", port=15000)},
            gpu_ids=[],
            worker_class=None,
        )
        worker_manager = _FakeWorkerManagerHandle(
            get_worker_infos=_FakeGetWorkerInfosMethod(controller_info=controller_info),
            get_actor_handle=_FakeGetActorHandleMethod(controller=controller),
        )
        monkeypatch.setattr(ray_worker_provider_mod.ray, "get", lambda value: value)
        capability = RayBackendCapability(worker_manager_handle=worker_manager)

        operations = capability.cell_operations()
        await operations.suspend(cell_id="cell-2")

        assert controller.stop_cell_between_weight_updates.suspended_cell_ids == ["cell-2"]
