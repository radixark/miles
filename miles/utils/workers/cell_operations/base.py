from __future__ import annotations

import abc

from miles.utils.test_utils.fault_injector.controller import FaultHookCommand
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, ObservedFaultHookTarget
from miles.utils.workers.worker_provider.base import CellInfo


class StaleFaultTargetError(Exception):
    pass


class BaseCellOperations(abc.ABC):
    @abc.abstractmethod
    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]: ...

    @abc.abstractmethod
    async def suspend(self, *, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def resume(self, *, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def observe_fault_target(self, *, cell_id: str, rank: int) -> ObservedFaultHookTarget: ...

    @abc.abstractmethod
    async def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord: ...
