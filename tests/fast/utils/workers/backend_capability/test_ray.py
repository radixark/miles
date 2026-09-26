from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import miles.utils.workers.backend_capability.ray as backend_capability_ray_mod
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.cell_operations.ray import RayCellOperations


@dataclass
class _RecordingRemoteMethod:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    async def remote(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


@dataclass
class _FakeWorkerManagerHandle:
    stop_cells: _RecordingRemoteMethod = field(default_factory=_RecordingRemoteMethod)
    start_cells: _RecordingRemoteMethod = field(default_factory=_RecordingRemoteMethod)
    inject_fault: _RecordingRemoteMethod = field(default_factory=_RecordingRemoteMethod)
    get_cell_infos: _RecordingRemoteMethod = field(default_factory=_RecordingRemoteMethod)


class TestRayBackendCapabilityCellOperations:
    async def test_suspend_reaches_the_worker_manager_directly(self) -> None:
        """Nothing may route a suspend through the inference controller, whose lock a weight update holds."""
        worker_manager = _FakeWorkerManagerHandle()
        capability = RayBackendCapability(worker_manager_handle=worker_manager)

        await capability.cell_operations().suspend(cell_id="cell-2")

        assert worker_manager.stop_cells.calls == [((["cell-2"],), {})]

    async def test_a_fault_reaches_the_worker_manager_directly(self) -> None:
        """Fault injection took the same controller detour as suspend, and must take the same direct path now."""
        worker_manager = _FakeWorkerManagerHandle()
        capability = RayBackendCapability(worker_manager_handle=worker_manager)

        await capability.cell_operations().inject_fault(cell_id="cell-2", mode=FailureMode.SIGKILL, sub_index=1)

        assert worker_manager.inject_fault.calls == [(("cell-2",), {"mode": "sigkill", "worker_in_cell_index": 1})]

    async def test_a_resume_reaches_the_worker_manager_directly(self) -> None:
        """Resume never went through the controller, and the shared path must not have changed it."""
        worker_manager = _FakeWorkerManagerHandle()
        capability = RayBackendCapability(worker_manager_handle=worker_manager)

        await capability.cell_operations().resume(cell_id="cell-2")

        assert worker_manager.start_cells.calls == [((["cell-2"],), {})]

    def test_the_capability_builds_a_ray_cell_operations_bound_to_its_own_handle(self) -> None:
        """A capability that handed out operations wired to anything else would heal the wrong fleet."""
        worker_manager = _FakeWorkerManagerHandle()

        operations = RayBackendCapability(worker_manager_handle=worker_manager).cell_operations()

        assert isinstance(operations, RayCellOperations)
        assert operations._worker_manager_handle is worker_manager

    def test_building_cell_operations_needs_no_inference_controller(self) -> None:
        """Resolving one at construction time is what forced this layer to import miles.ray."""
        worker_manager = _FakeWorkerManagerHandle()

        first = RayBackendCapability(worker_manager_handle=worker_manager).cell_operations()
        second = RayBackendCapability(worker_manager_handle=worker_manager).cell_operations()

        assert first is not second
        assert list(vars(first)) == ["_worker_manager_handle"]

    def test_the_module_does_not_import_the_ray_application_layer(self) -> None:
        """The deliberate layering violation existed only to resolve the inference controller."""
        source = inspect.getsource(backend_capability_ray_mod)

        assert "miles.ray.specs" not in source
        assert "create_inference_controller_handle" not in source
        assert not hasattr(backend_capability_ray_mod, "create_inference_controller_handle")
