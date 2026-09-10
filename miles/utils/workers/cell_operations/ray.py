from __future__ import annotations

import asyncio

import ray.actor

from miles.utils.test_utils.fault_hooks import FaultHookCommand, FaultHookRecord
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import (
    CELL_TERMINATION_NOT_CONFIRMED,
    TERMINATE_INCARNATION_TIMEOUT_SECONDS,
    BaseCellOperations,
    CellTerminationNotConfirmedError,
    CellTerminationOutcome,
    FaultTarget,
)
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

    async def observe_fault_target(self, *, cell_id: str, sub_index: int) -> FaultTarget:
        return await self._worker_manager_handle.observe_fault_target.remote(cell_id, sub_index=sub_index)

    async def control_fault_hook(self, *, target: FaultTarget, command: FaultHookCommand) -> str | FaultHookRecord:
        return await asyncio.wait_for(
            self._worker_manager_handle.control_fault_hook.remote(target=target, command=command), timeout=10.0
        )

    async def inject_fault(
        self,
        *,
        cell_id: str,
        mode: FailureMode,
        sub_index: int,
        expected_target: FaultTarget | None = None,
        request_id: str | None = None,
        receipt_url: str | None = None,
    ) -> None:
        if request_id is not None:
            await self._worker_manager_handle.inject_fault.remote(
                cell_id,
                mode=mode.value,
                worker_in_cell_index=sub_index,
                expected_target=expected_target,
                request_id=request_id,
                **({"receipt_url": receipt_url} if receipt_url is not None else {}),
            )
        elif expected_target is None:
            await self._worker_manager_handle.inject_fault.remote(
                cell_id, mode=mode.value, worker_in_cell_index=sub_index
            )
        else:
            await self._worker_manager_handle.inject_fault.remote(
                cell_id, mode=mode.value, worker_in_cell_index=sub_index, expected_target=expected_target
            )


async def _stop_cell_incarnation(
    worker_manager_handle: ray.actor.ActorHandle, *, cell_id: str, expected_workers_hash: str
) -> str:
    return await worker_manager_handle.stop_cell_incarnation.remote(
        cell_id, expected_workers_hash=expected_workers_hash
    )
