from __future__ import annotations

import abc

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.worker_provider.base import CellInfo


ROLLOUT_CELL_TYPE: str = "rollout"
ACTOR_CELL_TYPE: str = "actor"


class BaseCellOperations(abc.ABC):
    @abc.abstractmethod
    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]: ...

    @abc.abstractmethod
    async def suspend(self, *, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def resume(self, *, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def inject_fault(self, *, cell_id: str, cell_type: str, mode: FailureMode, sub_index: int) -> None: ...
