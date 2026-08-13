import abc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from miles.utils.workers.naming import compute_cell_id, parse_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.utils import build_rpc_handle_of_worker_info
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
    async def init(self) -> None:
        return None

    @abc.abstractmethod
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts: ...

    @abc.abstractmethod
    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]: ...

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        raise NotImplementedError(f"{type(self).__name__} answers addresses, it does not observe cells")

    def expected_num_cells(self, *, model_id: str) -> int | None:
        return None

    def get_handle(self, worker_name: str) -> BaseWorkerHandle:
        pool_id, cell_index, _worker_in_cell_index = parse_worker_name(worker_name)
        cell_id = compute_cell_id(pool_id=pool_id, cell_index=cell_index)
        (infos,) = self.get_worker_infos(cell_ids=[cell_id])
        handles = self.get_handles_of_worker_infos(infos)
        assert worker_name in handles, f"{worker_name=} is not one of {sorted(handles)}"
        return handles[worker_name]

    def get_handles_of_worker_infos(self, infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
        return {info.name: handle for info in infos if (handle := self._build_handle_of_worker_info(info)) is not None}

    def _build_handle_of_worker_info(self, info: WorkerInfo) -> BaseWorkerHandle | None:
        return build_rpc_handle_of_worker_info(info) if info.worker_class is not None else None
