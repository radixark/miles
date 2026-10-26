from __future__ import annotations

import asyncio

import ray.actor

from miles.utils.test_utils.fault_injector.controller import FaultHookCommand
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, ObservedFaultHookTarget
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_provider.base import CellInfo

CONTROL_FAULT_HOOK_TIMEOUT_SECONDS = 10.0


class RayCellOperations(BaseCellOperations):
    def __init__(self, *, worker_manager_handle: ray.actor.ActorHandle) -> None:
        self._worker_manager_handle = worker_manager_handle

    async def cell_infos(self, *, pool_ids: list[str] | None, category: str | None) -> dict[str, CellInfo]:
        return await self._worker_manager_handle.get_cell_infos.remote(pool_ids=pool_ids, category=category)

    async def suspend(self, *, cell_id: str) -> None:
        await self._worker_manager_handle.stop_cells.remote([cell_id])

    async def resume(self, *, cell_id: str) -> None:
        await self._worker_manager_handle.start_cells.remote([cell_id])

    async def observe_fault_target(self, *, cell_id: str, rank: int) -> ObservedFaultHookTarget:
        return await self._worker_manager_handle.observe_fault_target.remote(cell_id, rank=rank)

    async def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord:
        return await asyncio.wait_for(
            self._worker_manager_handle.control_fault_hook.remote(command=command),
            timeout=CONTROL_FAULT_HOOK_TIMEOUT_SECONDS,
        )
