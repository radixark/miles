import abc
from collections.abc import Awaitable, Callable
from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.naming import cell_id_of_worker
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.utils import build_rpc_handle_of_worker_info
from miles.utils.workers.worker_spec import NamedHostAndPorts


class CellInfo(FrozenStrictBaseModel):
    cell_id: str
    pool_id: str
    alive: bool
    worker_names: list[str]
    workers_hash: str
    meta: dict[str, Any]  # TODO: in k8s native mode, may be provided from pod annotations


# args: (cell_id, CellInfo)
CellReconcileFn = Callable[[str, CellInfo | None], Awaitable[None]]
StopWatchFn = Callable[[], Awaitable[None]]


class BaseWorkerProvider(abc.ABC):
    async def init(self) -> None:
        return None

    @abc.abstractmethod
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts: ...

    @abc.abstractmethod
    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]: ...

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        raise NotImplementedError(f"{type(self).__name__} answers addresses, it does not observe cells")

    def expected_num_cells(self, *, group_id: str) -> int | None:
        return None

    def get_handle(self, worker_name: str) -> BaseWorkerHandle:
        (infos,) = self.get_worker_infos(cell_ids=[cell_id_of_worker(worker_name)])
        handles = self.get_handles_of_worker_infos(infos)
        assert worker_name in handles, f"{worker_name=} is not one of {sorted(handles)}"
        return handles[worker_name]

    def get_handles_of_worker_infos(self, infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
        return {info.name: handle for info in infos if (handle := self._build_handle_of_worker_info(info)) is not None}

    def _build_handle_of_worker_info(self, info: WorkerInfo) -> BaseWorkerHandle | None:
        return build_rpc_handle_of_worker_info(info) if info.worker_class is not None else None
