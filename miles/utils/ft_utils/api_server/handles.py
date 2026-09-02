from __future__ import annotations

import asyncio
from typing import Protocol

from miles.ray.rollout.server_cell import compute_pending_rollout_cell_status
from miles.utils.ft_utils.api_server.models import Cell, CellCondition, CellMetadata, CellSpec, CellStatus, TriState
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_provider.base import CellInfo


class _CellStatusSource(Protocol):
    async def get_cell_statuses(self) -> dict[str, CellStatus]: ...


class _CellHandler:
    def __init__(
        self,
        *,
        cell_type: str,
        operations: BaseCellOperations,
        controllers: list[_CellStatusSource],
        pool_ids: list[str],
    ) -> None:
        self._cell_type = cell_type
        self._operations = operations
        self._controllers = controllers
        self._pool_ids = pool_ids

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
        statuses = await self._get_cell_statuses()
        return [
            self._compute_cell(cell_id, cell_infos=cell_infos, statuses=statuses) for cell_id in sorted(cell_infos)
        ]

    async def get_cell(self, cell_id: str) -> Cell:
        return self._compute_cell(
            cell_id,
            cell_infos=await self._get_cell_infos(),
            statuses=await self._get_cell_statuses(),
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

    async def _get_cell_statuses(self) -> dict[str, CellStatus]:
        return {
            cell_id: status
            for statuses in await asyncio.gather(*(c.get_cell_statuses() for c in self._controllers))
            for cell_id, status in statuses.items()
        }

    async def _get_cell_infos(self) -> dict[str, CellInfo]:
        return await self._operations.cell_infos(pool_ids=self._pool_ids)

    async def suspend(self, cell_id: str) -> None:
        await self._operations.suspend(cell_id=cell_id)

    async def resume(self, cell_id: str) -> None:
        await self._operations.resume(cell_id=cell_id)

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        await self._operations.inject_fault(cell_id=cell_id, cell_type=self._cell_type, mode=mode, sub_index=sub_index)


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
