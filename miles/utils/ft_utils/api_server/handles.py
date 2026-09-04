from __future__ import annotations

import abc

from miles.ray.train.group import RayTrainGroup
from miles.utils.ft_utils.api_server.models import Cell, CellCondition, CellMetadata, CellSpec, CellStatus
from miles.utils.test_utils.fault_injector import FailureMode


class _CellHandle(abc.ABC):
    @property
    def cell_id(self) -> str:
        return f"{self.cell_type}-{self.cell_key}"

    @property
    @abc.abstractmethod
    def cell_type(self) -> str: ...

    @property
    @abc.abstractmethod
    def cell_key(self) -> str: ...

    @abc.abstractmethod
    async def get_cell(self) -> Cell: ...

    @abc.abstractmethod
    async def suspend(self) -> None: ...

    @abc.abstractmethod
    async def resume(self) -> None: ...

    async def inject_fault(self, *, mode: FailureMode, sub_index: int) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support fault injection")


class _ActorCellHandle(_CellHandle):
    def __init__(self, *, group: RayTrainGroup, cell_index: int) -> None:
        self._group = group
        self._cell_index = cell_index

    @property
    def cell_type(self) -> str:
        return "actor"

    @property
    def cell_key(self) -> str:
        return str(self._cell_index)

    async def get_cell(self) -> Cell:
        cell = self._group._cells[self._cell_index]
        return Cell(
            metadata=CellMetadata(
                name=self.cell_id,
                labels={
                    "miles.io/cell-type": "actor",
                    "miles.io/cell-index": self.cell_key,
                },
            ),
            spec=CellSpec(suspend=cell.is_stopped),
            status=cell.cell_status(),
        )

    async def suspend(self) -> None:
        self._group.stop_cell(self._cell_index)

    async def resume(self) -> None:
        self._group.start_cell(self._cell_index)

    async def inject_fault(self, *, mode: FailureMode, sub_index: int) -> None:
        """Inject a fault into a specific actor of this cell. Fire-and-forget."""
        cell = self._group._cells[self._cell_index]
        if not cell.is_alive:
            raise RuntimeError(f"Cell {self._cell_index} is not alive, cannot inject fault")
        actors = cell._get_actor_handles()
        if sub_index < 0 or sub_index >= len(actors):
            raise IndexError(
                f"sub_index {sub_index} out of range for cell {self._cell_index} " f"(has {len(actors)} actors)"
            )
        actors[sub_index].inject_fault.remote(mode.value)


# TODO the code will NOT work before implementing rollout ft
class _RolloutCellHandle(_CellHandle):
    def __init__(self, *, inference_controller: object, rollout_cell_id: str) -> None:
        self._inference_controller = inference_controller
        self._rollout_cell_id = rollout_cell_id

    @property
    def cell_type(self) -> str:
        return "rollout"

    @property
    def cell_key(self) -> str:
        return self._rollout_cell_id

    async def get_cell(self) -> Cell:
        phase = self._inference_controller.get_cell_phase(self._rollout_cell_id)
        conditions_raw = self._inference_controller.get_cell_conditions(self._rollout_cell_id)
        is_suspended = self._inference_controller.get_cell_is_suspended(self._rollout_cell_id)
        return Cell(
            metadata=CellMetadata(
                name=self.cell_id,
                labels={
                    "miles.io/cell-type": "rollout",
                    "miles.io/cell-index": self.cell_key,
                },
            ),
            spec=CellSpec(suspend=is_suspended),
            status=CellStatus(
                phase=phase,
                conditions=[CellCondition(**c) for c in conditions_raw],
            ),
        )

    async def suspend(self) -> None:
        await self._inference_controller.stop_cell(self._rollout_cell_id)

    async def resume(self) -> None:
        await self._inference_controller.start_cell(self._rollout_cell_id)
