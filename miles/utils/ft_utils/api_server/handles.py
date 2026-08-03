from __future__ import annotations

import abc
import asyncio

import ray

from miles.ray.rollout.server_cell import PENDING_ROLLOUT_CELL_STATUS
from miles.ray.train.group import RayTrainGroup
from miles.utils.ft_utils.api_server.models import Cell, CellCondition, CellMetadata, CellSpec, CellStatus, TriState
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.worker_provider.base import CellInfo


_ACTOR_CELL_ID_PREFIX = "actor-"


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


def _compute_actor_cell_id(cell_index: int) -> str:
    return f"{_ACTOR_CELL_ID_PREFIX}{cell_index}"


def _parse_actor_cell_index(cell_id: str) -> int:
    assert cell_id.startswith(_ACTOR_CELL_ID_PREFIX), f"{cell_id=}"
    return int(cell_id[len(_ACTOR_CELL_ID_PREFIX) :])


class _ActorCellHandler(_CellHandler):
    def __init__(self, *, group: RayTrainGroup) -> None:
        self._group = group

    @property
    def cell_type(self) -> str:
        return "actor"

    async def list_cell_ids(self) -> list[str]:
        return [_compute_actor_cell_id(cell_index) for cell_index in range(len(self._group._cells))]

    async def get_cell(self, cell_id: str) -> Cell:
        cell = self._find_cell(cell_id)
        return Cell(
            metadata=self._compute_metadata(cell_id),
            spec=CellSpec(suspend=cell.is_stopped),
            status=cell.cell_status(),
        )

    async def suspend(self, cell_id: str) -> None:
        self._group.stop_cell(_parse_actor_cell_index(cell_id))

    async def resume(self, cell_id: str) -> None:
        self._group.start_cell(_parse_actor_cell_index(cell_id))

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        cell = self._find_cell(cell_id)
        if not cell.is_alive:
            raise RuntimeError(f"Cell {cell_id} is not alive, cannot inject fault")
        actors = cell._get_actor_handles()
        if sub_index < 0 or sub_index >= len(actors):
            raise IndexError(f"sub_index {sub_index} out of range for cell {cell_id} (has {len(actors)} actors)")
        actors[sub_index].inject_fault.remote(mode.value)

    def _find_cell(self, cell_id: str):
        return self._group._cells[_parse_actor_cell_index(cell_id)]


class _RolloutCellHandler(_CellHandler):
    def __init__(
        self,
        *,
        worker_manager: ray.actor.ActorHandle,
        inference_controller: object,
        engine_spec_names: list[str],
    ) -> None:
        self._worker_manager = worker_manager
        self._inference_controller = inference_controller
        self._engine_spec_names = engine_spec_names

    @property
    def cell_type(self) -> str:
        return "rollout"

    async def list_cell_ids(self) -> list[str]:
        return sorted(await self._get_cell_infos())

    async def list_cells(self) -> list[Cell]:
        cell_infos = await self._get_cell_infos()
        statuses = self._inference_controller.get_cell_statuses()
        return [
            self._compute_cell(cell_id, cell_infos=cell_infos, statuses=statuses) for cell_id in sorted(cell_infos)
        ]

    async def get_cell(self, cell_id: str) -> Cell:
        return self._compute_cell(
            cell_id,
            cell_infos=await self._get_cell_infos(),
            statuses=self._inference_controller.get_cell_statuses(),
        )

    def _compute_cell(self, cell_id: str, *, cell_infos: dict[str, CellInfo], statuses: dict[str, CellStatus]) -> Cell:
        suspended = not cell_infos[cell_id].alive
        return Cell(
            metadata=self._compute_metadata(cell_id),
            spec=CellSpec(suspend=suspended),
            status=(
                CellStatus(phase="Suspended", conditions=[CellCondition.allocated(TriState.FALSE)])
                if suspended
                else statuses.get(cell_id, PENDING_ROLLOUT_CELL_STATUS)
            ),
        )

    async def _get_cell_infos(self) -> dict[str, CellInfo]:
        return await self._worker_manager.get_cell_infos.remote(spec_names=self._engine_spec_names)

    async def suspend(self, cell_id: str) -> None:
        await self._worker_manager.stop_cells.remote([cell_id])

    async def resume(self, cell_id: str) -> None:
        await self._worker_manager.start_cells.remote([cell_id])

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        await self._worker_manager.inject_fault.remote(cell_id, mode=mode.value, worker_in_cell_index=sub_index)
