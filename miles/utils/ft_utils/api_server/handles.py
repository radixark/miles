from __future__ import annotations

import abc
import asyncio

import ray

from miles.ray.rollout.server_cell import compute_pending_rollout_cell_status
from miles.ray.train.group import RayTrainGroup
from miles.utils.ft_utils.api_server.models import Cell, CellCondition, CellMetadata, CellSpec, CellStatus, TriState
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.naming import parse_cell_id
from miles.utils.workers.worker_provider.base import CellInfo


class _CellHandler(abc.ABC):
    @property
    @abc.abstractmethod
    def cell_type(self) -> str: ...

    @abc.abstractmethod
    async def list_cell_ids(self) -> list[str]: ...

    @abc.abstractmethod
    async def get_cell(self, cell_id: str) -> Cell: ...

    @abc.abstractmethod
    async def suspend(self, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def resume(self, cell_id: str) -> None: ...

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support fault injection")

    async def list_cells(self) -> list[Cell]:
        cell_ids = await self.list_cell_ids()
        return list(await asyncio.gather(*(self.get_cell(cell_id) for cell_id in cell_ids)))

    def _compute_metadata(self, cell_id: str) -> CellMetadata:
        return CellMetadata(
            name=cell_id,
            labels={
                "miles.io/cell-type": self.cell_type,
                "miles.io/cell-id": cell_id,
            },
        )


class _ActorCellHandler(_CellHandler):
    def __init__(
        self,
        *,
        worker_manager: ray.actor.ActorHandle,
        group: RayTrainGroup,
        trainer_pool_ids: list[str],
    ) -> None:
        self._worker_manager = worker_manager
        self._group = group
        self._trainer_pool_ids = trainer_pool_ids

    @property
    def cell_type(self) -> str:
        return "actor"

    async def list_cell_ids(self) -> list[str]:
        return sorted(await self._get_cell_infos())

    async def list_cells(self) -> list[Cell]:
        cell_infos = await self._get_cell_infos()
        statuses = self._group.get_cell_statuses()
        return [
            self._compute_cell(cell_id, cell_infos=cell_infos, statuses=statuses) for cell_id in sorted(cell_infos)
        ]

    async def get_cell(self, cell_id: str) -> Cell:
        return self._compute_cell(
            cell_id,
            cell_infos=await self._get_cell_infos(),
            statuses=self._group.get_cell_statuses(),
        )

    def _compute_cell(self, cell_id: str, *, cell_infos: dict[str, CellInfo], statuses: dict[str, CellStatus]) -> Cell:
        status = statuses.get(cell_id)
        if not cell_infos[cell_id].alive and (status is None or status.phase != "Pending"):
            status = CellStatus(phase="Suspended", conditions=[CellCondition.allocated(TriState.FALSE)])
        elif status is None:
            status = compute_pending_rollout_cell_status()
        return Cell(
            metadata=self._compute_metadata(cell_id),
            spec=CellSpec(suspend=status.phase == "Suspended"),
            status=status,
        )

    async def _get_cell_infos(self) -> dict[str, CellInfo]:
        return await self._worker_manager.get_cell_infos.remote(pool_ids=self._trainer_pool_ids)

    async def suspend(self, cell_id: str) -> None:
        await self._group.stop_cell(parse_cell_id(cell_id).cell_index)

    async def resume(self, cell_id: str) -> None:
        self._group.start_cell(parse_cell_id(cell_id).cell_index)

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        await self._worker_manager.inject_fault.remote(cell_id, mode=mode.value, worker_in_cell_index=sub_index)


class _RolloutCellHandler(_CellHandler):
    def __init__(
        self,
        *,
        worker_manager: ray.actor.ActorHandle,
        inference_controller: object,
    ) -> None:
        self._worker_manager = worker_manager
        self._inference_controller = inference_controller

    @property
    def cell_type(self) -> str:
        return "rollout"

    async def list_cell_ids(self) -> list[str]:
        return sorted(self._inference_controller.get_cell_statuses())

    async def list_cells(self) -> list[Cell]:
        statuses = self._inference_controller.get_cell_statuses()
        return [self._compute_cell(cell_id, statuses=statuses) for cell_id in sorted(statuses)]

    async def get_cell(self, cell_id: str) -> Cell:
        return self._compute_cell(cell_id, statuses=self._inference_controller.get_cell_statuses())

    def _compute_cell(self, cell_id: str, *, statuses: dict[str, object]) -> Cell:
        status = statuses.get(cell_id) or compute_pending_rollout_cell_status()
        return Cell(
            metadata=self._compute_metadata(cell_id),
            spec=CellSpec(suspend=status.phase == "Suspended"),
            status=status,
        )

    async def suspend(self, cell_id: str) -> None:
        await self._worker_manager.stop_cells.remote([cell_id])
        self._inference_controller.notify_cell_suspended(cell_id)

    async def resume(self, cell_id: str) -> None:
        self._inference_controller.notify_cell_resumed(cell_id)
        await self._worker_manager.start_cells.remote([cell_id])

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        await self._worker_manager.inject_fault.remote(cell_id, mode=mode.value, worker_in_cell_index=sub_index)
