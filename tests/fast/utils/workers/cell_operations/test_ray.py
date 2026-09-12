from __future__ import annotations

from typing import Any

from miles.utils.workers.cell_operations.ray import RayCellOperations


class _RecordingRemoteMethod:
    def __init__(self, *, name: str, calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> None:
        self._name = name
        self._calls = calls
        self.result: dict[str, Any] = {}

    async def remote(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._calls.append((self._name, args, kwargs))
        return self.result


class _RecordingWorkerManagerHandle:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.get_cell_infos = _RecordingRemoteMethod(name="get_cell_infos", calls=self.calls)
        self.start_cells = _RecordingRemoteMethod(name="start_cells", calls=self.calls)


class TestRayCellOperationsProtocol:
    async def test_cell_infos_forwards_pool_ids_and_returns_the_actor_result(self) -> None:
        """Cell info reads forward every pool ID by keyword and preserve the actor result."""
        worker_manager = _RecordingWorkerManagerHandle()
        operations = RayCellOperations(worker_manager_handle=worker_manager)
        pool_ids = ["engine-0", "rollout-1"]
        actor_result = {"engine-0-2": object(), "rollout-1-3": object()}
        worker_manager.get_cell_infos.result = actor_result

        result = await operations.cell_infos(pool_ids=pool_ids)

        assert worker_manager.calls == [("get_cell_infos", (), {"pool_ids": pool_ids})]
        assert result is actor_result

    async def test_resume_starts_exactly_the_requested_cell(self) -> None:
        """Resuming a cell sends exactly that cell ID in a one-element list."""
        worker_manager = _RecordingWorkerManagerHandle()
        operations = RayCellOperations(worker_manager_handle=worker_manager)

        await operations.resume(cell_id="engine-0-2")

        assert worker_manager.calls == [("start_cells", (["engine-0-2"],), {})]
