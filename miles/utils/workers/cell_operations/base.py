from __future__ import annotations

import abc
import enum

from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.worker_provider.base import CellInfo

TERMINATE_INCARNATION_TIMEOUT_SECONDS = 120.0
CELL_TERMINATION_NOT_CONFIRMED = "not_confirmed"


class CellTerminationOutcome(enum.Enum):
    TERMINATED = "terminated"
    ALREADY_GONE = "already_gone"
    STALE = "stale"


class CellTerminationNotConfirmedError(Exception):
    pass


class StaleFaultTargetError(Exception):
    pass


class FaultTarget(FrozenStrictBaseModel):
    cell_id: str
    sub_index: int = Field(ge=0)
    workers_hash: str = Field(min_length=1)
    boot_uuid: str | None = None
    pod_uid: str | None = None


class BaseCellOperations(abc.ABC):
    @abc.abstractmethod
    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]: ...

    @abc.abstractmethod
    async def suspend(self, *, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def resume(self, *, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def terminate_incarnation(
        self,
        *,
        cell_id: str,
        expected_workers_hash: str,
        timeout: float = TERMINATE_INCARNATION_TIMEOUT_SECONDS,
    ) -> CellTerminationOutcome: ...

    @abc.abstractmethod
    async def inject_fault(
        self, *, cell_id: str, mode: FailureMode, sub_index: int, expected_target: FaultTarget | None = None
    ) -> None: ...

    async def observe_fault_target(self, *, cell_id: str, sub_index: int) -> FaultTarget:
        raise NotImplementedError("This backend does not expose fault target identities")
