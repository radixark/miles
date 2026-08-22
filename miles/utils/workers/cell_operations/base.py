from __future__ import annotations

import abc

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.worker_provider.base import CellInfo


class CellGenerationMismatchError(RuntimeError):
    pass


class BaseCellOperations(abc.ABC):
    @abc.abstractmethod
    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]: ...

    @abc.abstractmethod
    async def suspend(self, *, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def resume(self, *, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def inject_fault(
        self,
        *,
        cell_id: str,
        expected_workers_hash: str,
        mode: FailureMode,
        sub_index: int,
    ) -> None: ...
