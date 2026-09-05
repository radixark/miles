from __future__ import annotations

from collections.abc import Callable

import ray.actor

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import CellInfo


class RayCellOperations(BaseCellOperations):
    def __init__(
        self,
        *,
        worker_manager_handle: ray.actor.ActorHandle,
        resolve_inference_controller: Callable[[], BaseWorkerHandle],
    ) -> None:
        self._worker_manager_handle = worker_manager_handle
        # TEMPORARY: this layer is not meant to know the inference controller, deliberately violated
        # until the weight-update fault tolerance work removes the need
        self._resolve_inference_controller = resolve_inference_controller
        self._inference_controller: BaseWorkerHandle | None = None

    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        return await self._worker_manager_handle.get_cell_infos.remote(pool_ids=pool_ids)

    async def suspend(self, *, cell_id: str) -> None:
        # TEMPORARY: taking the lock the weight update holds, reverted with that fault tolerance work
        # await self._worker_manager_handle.stop_cells.remote([cell_id])  # use this later
        if self._inference_controller is None:
            self._inference_controller = self._resolve_inference_controller()
        await self._inference_controller.stop_cell_between_weight_updates(cell_id=cell_id)

    async def resume(self, *, cell_id: str) -> None:
        await self._worker_manager_handle.start_cells.remote([cell_id])

    async def inject_fault(self, *, cell_id: str, mode: FailureMode, sub_index: int) -> None:
        await self._worker_manager_handle.inject_fault.remote(cell_id, mode=mode.value, worker_in_cell_index=sub_index)
