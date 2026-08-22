from __future__ import annotations

import asyncio
from typing import Protocol

from miles.ray.rollout.server_cell import compute_pending_rollout_cell_status
from miles.utils.ft_utils.api_server.models import (
    WORKERS_HASH_LABEL,
    Cell,
    CellCondition,
    CellMetadata,
    CellSpec,
    CellStatus,
    CellStatusSnapshot,
    TriState,
)
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_provider.base import CellInfo


class _CellStatusSource(Protocol):
    async def get_cell_statuses(self) -> dict[str, CellStatus]: ...

    async def get_cell_status_snapshots(self) -> dict[str, CellStatusSnapshot]: ...


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

    def _compute_metadata(self, cell_id: str, *, workers_hash: str) -> CellMetadata:
        return CellMetadata(
            name=cell_id,
            labels={
                "miles.io/cell-type": self.cell_type,
                "miles.io/cell-id": cell_id,
                WORKERS_HASH_LABEL: workers_hash,
            },
        )

    async def list_cell_ids(self) -> list[str]:
        return sorted(await self._get_cell_infos())

    async def list_cells(self) -> list[Cell]:
        cell_infos = await self._get_cell_infos()
        snapshots = await self._get_cell_status_snapshots()
        return [
            self._compute_cell(cell_id, cell_infos=cell_infos, snapshots=snapshots) for cell_id in sorted(cell_infos)
        ]

    async def get_cell(self, cell_id: str) -> Cell:
        return self._compute_cell(
            cell_id,
            cell_infos=await self._get_cell_infos(),
            snapshots=await self._get_cell_status_snapshots(),
        )

    async def _get_cell_statuses(self) -> dict[str, CellStatus]:
        return {cell_id: snapshot.status for cell_id, snapshot in (await self._get_cell_status_snapshots()).items()}

    async def _get_cell_status_snapshots(self) -> dict[str, CellStatusSnapshot]:
        return {
            cell_id: snapshot
            for snapshots in await asyncio.gather(*(c.get_cell_status_snapshots() for c in self._controllers))
            for cell_id, snapshot in snapshots.items()
        }

    def _compute_cell(
        self,
        cell_id: str,
        *,
        cell_infos: dict[str, CellInfo],
        snapshots: dict[str, CellStatusSnapshot],
    ) -> Cell:
        cell_info = cell_infos[cell_id]
        suspended = not cell_info.alive
        snapshot = snapshots.get(cell_id)
        if snapshot is not None and snapshot.workers_hash != cell_info.workers_hash:
            raise RuntimeError(
                f"Cell {cell_id} changed workers generation while its status was being read: "
                f"{snapshot.workers_hash} != {cell_info.workers_hash}"
            )
        return Cell(
            metadata=self._compute_metadata(cell_id, workers_hash=cell_info.workers_hash),
            spec=CellSpec(suspend=suspended),
            status=(
                CellStatus(phase="Suspended", conditions=[CellCondition.allocated(TriState.FALSE)])
                if suspended
                else snapshot.status if snapshot is not None else compute_pending_rollout_cell_status()
            ),
        )

    async def _get_cell_infos(self) -> dict[str, CellInfo]:
        return await self._operations.cell_infos(pool_ids=self._pool_ids)

    async def suspend(self, cell_id: str) -> None:
        await self._operations.suspend(cell_id=cell_id)

    async def resume(self, cell_id: str) -> None:
        await self._operations.resume(cell_id=cell_id)

    async def inject_fault(
        self,
        cell_id: str,
        *,
        expected_workers_hash: str,
        mode: FailureMode,
        sub_index: int,
    ) -> None:
        await self._operations.inject_fault(
            cell_id=cell_id,
            expected_workers_hash=expected_workers_hash,
            mode=mode,
            sub_index=sub_index,
        )
