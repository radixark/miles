from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, Protocol

import ray

from miles.ray.rollout.server_cell import compute_pending_rollout_cell_status
from miles.utils.ft_utils.api_server.models import Cell, CellCondition, CellMetadata, CellSpec, CellStatus, TriState
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.worker_provider.base import CellInfo


class _CellStatusSource(Protocol):
    async def get_cell_statuses(self) -> dict[str, CellStatus]: ...


# TEMPORARY: this layer is not meant to know the inference controller, deliberately violated
# until the weight-update fault tolerance work removes the need
class _CellOperations(Protocol):
    async def stop_cell_between_weight_updates(self, cell_id: str) -> None: ...

    async def inject_fault_between_weight_updates(
        self, cell_id: str, *, mode: FailureMode, sub_index: int
    ) -> None: ...


class _CellHandler:
    def __init__(
        self,
        *,
        cell_type: str,
        worker_manager: ray.actor.ActorHandle,
        controller: _CellStatusSource,
        pool_ids: list[str],
        cell_operations: _CellOperations | None = None,
        cell_operations_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        assert (cell_operations is None) == (
            cell_operations_loop is None
        ), "cell operations only run on the loop that owns them, so both must be given together"
        self._cell_type = cell_type
        self._worker_manager = worker_manager
        self._controller = controller
        self._pool_ids = pool_ids
        self._cell_operations = cell_operations
        self._cell_operations_loop = cell_operations_loop

    @property
    def cell_type(self) -> str:
        return self._cell_type

    def _compute_metadata(self, cell_id: str) -> CellMetadata:
        return CellMetadata(
            name=cell_id,
            labels={
                "miles.io/cell-type": self.cell_type,
                "miles.io/cell-id": cell_id,
            },
        )

    async def list_cell_ids(self) -> list[str]:
        return sorted(await self._get_cell_infos())

    async def list_cells(self) -> list[Cell]:
        cell_infos = await self._get_cell_infos()
        statuses = await self._controller.get_cell_statuses()
        return [
            self._compute_cell(cell_id, cell_infos=cell_infos, statuses=statuses) for cell_id in sorted(cell_infos)
        ]

    async def get_cell(self, cell_id: str) -> Cell:
        return self._compute_cell(
            cell_id,
            cell_infos=await self._get_cell_infos(),
            statuses=await self._controller.get_cell_statuses(),
        )

    def _compute_cell(self, cell_id: str, *, cell_infos: dict[str, CellInfo], statuses: dict[str, CellStatus]) -> Cell:
        info = cell_infos[cell_id]
        suspended = not info.alive
        return Cell(
            metadata=self._compute_metadata(cell_id),
            spec=CellSpec(suspend=suspended),
            status=(
                CellStatus(
                    phase="Suspended",
                    conditions=[CellCondition.allocated(TriState.FALSE)],
                    workers_hash=info.workers_hash,
                )
                if suspended
                else _compute_status_of_generation(statuses.get(cell_id), workers_hash=info.workers_hash)
            ),
        )

    async def _get_cell_infos(self) -> dict[str, CellInfo]:
        return await self._worker_manager.get_cell_infos.remote(pool_ids=self._pool_ids)

    async def suspend(self, cell_id: str) -> None:
        # TEMPORARY: taking the lock the weight update holds, reverted with that fault tolerance work
        if (operations := self._cell_operations) is not None:
            await self._run_on_cell_operations_loop(operations.stop_cell_between_weight_updates(cell_id=cell_id))
            return
        await self._worker_manager.stop_cells.remote([cell_id])

    async def resume(self, cell_id: str) -> None:
        await self._worker_manager.start_cells.remote([cell_id])

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        # TEMPORARY: taking the lock the weight update holds, reverted with that fault tolerance work
        if (operations := self._cell_operations) is not None:
            await self._run_on_cell_operations_loop(
                operations.inject_fault_between_weight_updates(cell_id=cell_id, mode=mode, sub_index=sub_index)
            )
            return
        await self._worker_manager.inject_fault.remote(cell_id, mode=mode.value, worker_in_cell_index=sub_index)

    async def _run_on_cell_operations_loop(self, coroutine: Coroutine[Any, Any, None]) -> None:
        loop = self._cell_operations_loop
        assert loop is not None
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coroutine, loop))


def _compute_status_of_generation(status: CellStatus | None, *, workers_hash: str) -> CellStatus:
    if status is None:
        return compute_pending_rollout_cell_status(workers_hash=workers_hash)
    if status.workers_hash != workers_hash:
        return status.model_copy(
            update={
                "conditions": [_compute_unknown_condition(condition) for condition in status.conditions],
                "workers_hash": workers_hash,
            }
        )
    return status


def _compute_unknown_condition(condition: CellCondition) -> CellCondition:
    if condition.type == "Healthy":
        return CellCondition.from_health_checker_status(TriState.UNKNOWN)
    return CellCondition(type=condition.type, status=TriState.UNKNOWN)
