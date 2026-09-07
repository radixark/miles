from __future__ import annotations

import asyncio
from collections.abc import Callable

import ray.actor

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import (
    CELL_TERMINATION_NOT_CONFIRMED,
    TERMINATE_INCARNATION_TIMEOUT_SECONDS,
    BaseCellOperations,
    CellTerminationNotConfirmedError,
    CellTerminationOutcome,
)
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
        if _is_trainer_cell_id(cell_id):
            await self._worker_manager_handle.stop_cells.remote([cell_id])
            return

        # TEMPORARY: taking the lock the weight update holds, reverted with that fault tolerance work
        await self._controller().stop_cell_between_weight_updates(cell_id=cell_id)

    async def resume(self, *, cell_id: str) -> None:
        await self._worker_manager_handle.start_cells.remote([cell_id])

    async def terminate_incarnation(
        self,
        *,
        cell_id: str,
        expected_workers_hash: str,
        timeout: float = TERMINATE_INCARNATION_TIMEOUT_SECONDS,
    ) -> CellTerminationOutcome:
        try:
            outcome = await asyncio.wait_for(
                _stop_cell_incarnation(
                    self._worker_manager_handle, cell_id=cell_id, expected_workers_hash=expected_workers_hash
                ),
                timeout=timeout,
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise CellTerminationNotConfirmedError(
                f"the worker manager did not answer within {timeout}s whether it stopped {cell_id} "
                f"({expected_workers_hash}), so its workers may still be running"
            ) from e
        if outcome == CELL_TERMINATION_NOT_CONFIRMED:
            raise CellTerminationNotConfirmedError(
                f"the worker manager killed {cell_id} ({expected_workers_hash}) but its actors kept answering, "
                f"so their workers may still be running"
            )
        return CellTerminationOutcome(outcome)

    async def inject_fault(self, *, cell_id: str, mode: FailureMode, sub_index: int) -> None:
        if _is_trainer_cell_id(cell_id):
            await self._worker_manager_handle.inject_fault.remote(
                cell_id, mode=mode.value, worker_in_cell_index=sub_index
            )
            return

        # TEMPORARY: taking the lock the weight update holds, reverted with that fault tolerance work
        await self._controller().inject_fault_between_weight_updates(
            cell_id=cell_id,
            mode=mode,
            sub_index=sub_index,
        )

    def _controller(self) -> BaseWorkerHandle:
        if self._inference_controller is None:
            self._inference_controller = self._resolve_inference_controller()
        return self._inference_controller


async def _stop_cell_incarnation(
    worker_manager_handle: ray.actor.ActorHandle, *, cell_id: str, expected_workers_hash: str
) -> str:
    return await worker_manager_handle.stop_cell_incarnation.remote(
        cell_id, expected_workers_hash=expected_workers_hash
    )


# TEMPORARY: matched by pool prefix only, until the fault tolerance rework routes trainer cells properly
def _is_trainer_cell_id(cell_id: str) -> bool:
    return cell_id.startswith("trainer-engine-")
