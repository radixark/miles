import abc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from miles.utils.workers.naming import compute_cell_id, parse_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
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

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        raise NotImplementedError(f"{type(self).__name__} answers addresses, it does not observe cells")

    def get_handle(self, worker_name: str) -> BaseWorkerHandle:
        pool_id, cell_index, _worker_in_cell_index = parse_worker_name(worker_name)
        cell_id = compute_cell_id(pool_id=pool_id, cell_index=cell_index)
        (infos,) = self.get_worker_infos(cell_ids=[cell_id])
        matches = [info for info in infos if info.name == worker_name]
        assert len(matches) == 1, f"{worker_name=} matched {[info.name for info in matches]}"
        handle = matches[0].handle
        assert handle is not None, f"pool {pool_id} has no worker class, so its rpc methods are unknown"
        return handle
