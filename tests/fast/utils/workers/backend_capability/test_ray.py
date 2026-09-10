from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from miles.utils.workers.backend_capability.ray import RayBackendCapability


@dataclass
class _RecordingRemoteMethod:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    async def remote(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


@dataclass
class _FakeWorkerManagerHandle:
    stop_cells: _RecordingRemoteMethod = field(default_factory=_RecordingRemoteMethod)


class TestRayBackendCapabilityCellOperations:
    async def test_suspend_reaches_the_worker_manager_directly(self) -> None:
        """Nothing may route a suspend through the inference controller, whose lock a weight update holds."""
        worker_manager = _FakeWorkerManagerHandle()
        capability = RayBackendCapability(worker_manager_handle=worker_manager)

        await capability.cell_operations().suspend(cell_id="cell-2")

        assert worker_manager.stop_cells.calls == [((["cell-2"],), {})]
