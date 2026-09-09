import abc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_spec import NamedHostAndPorts


@dataclass(frozen=True)
class CellInfo:
    cell_id: str
    pool_id: str
    alive: bool
    worker_names: list[str]
    workers_hash: str
    meta: dict[str, Any]  # TODO: in k8s native mode, may be provided from pod annotations


# args: (cell_id, CellInfo)
ReconcileFn = Callable[[str, CellInfo | None], Awaitable[None]]
StopWatchFn = Callable[[], Awaitable[None]]


class BaseWorkerProvider(abc.ABC):
    @abc.abstractmethod
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts: ...

    @abc.abstractmethod
    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]: ...

    @abc.abstractmethod
    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn: ...
