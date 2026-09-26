from __future__ import annotations

import ray.actor

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_provider.base import CellInfo


class RayCellOperations(BaseCellOperations):
    def __init__(self, *, worker_manager_handle: ray.actor.ActorHandle) -> None:
        self._worker_manager_handle = worker_manager_handle

    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        return await self._worker_manager_handle.get_cell_infos.remote(pool_ids=pool_ids)

    async def suspend(self, *, cell_id: str) -> None:
        await self._worker_manager_handle.stop_cells.remote([cell_id])

    async def resume(self, *, cell_id: str) -> None:
        await self._worker_manager_handle.start_cells.remote([cell_id])

    async def inject_fault(self, *, cell_id: str, mode: FailureMode, sub_index: int) -> None:
        await self._worker_manager_handle.inject_fault.remote(cell_id, mode=mode.value, worker_in_cell_index=sub_index)
