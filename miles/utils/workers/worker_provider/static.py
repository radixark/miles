from __future__ import annotations

from miles.utils.function_registry import load_function
from miles.utils.workers.naming import parse_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm import naming
from miles.utils.workers.worker_provider.utils import build_rpc_handle
from miles.utils.workers.worker_spec import BaseWorkerSpec, NamedHostAndPorts, ServeWorkerSpec


class StaticWorkerProvider(BaseWorkerProvider):
    def __init__(self, *, release: str, spec: BaseWorkerSpec) -> None:
        self._release = release
        self._spec = spec

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return self._addrs_of_worker(worker_name)

    def get_handle(self, worker_name: str) -> BaseWorkerHandle:
        assert isinstance(
            self._spec, ServeWorkerSpec
        ), f"pool {self._spec.name} is launched as a command rather than served, so its rpc methods are unknown"
        return build_rpc_handle(
            worker_class=load_function(self._spec.worker_class), addrs=self._addrs_of_worker(worker_name)
        )

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        raise NotImplementedError(f"{type(self).__name__} answers addresses, it does not enumerate workers")

    def _addrs_of_worker(self, worker_name: str) -> NamedHostAndPorts:
        scheduling = self._spec.scheduling
        pool_id, cell_index, worker_in_cell_index = parse_worker_name(worker_name)
        assert pool_id == self._spec.name, f"this provider answers for pool {self._spec.name}, not {pool_id}"
        assert (
            cell_index < scheduling.num_cells
        ), f"pool {pool_id} deploys {scheduling.num_cells} cells, so cell {cell_index} is not one of them"
        assert worker_in_cell_index < scheduling.num_workers_per_cell, (
            f"a cell of pool {pool_id} holds {scheduling.num_workers_per_cell} workers, "
            f"so worker {worker_in_cell_index} is not one of them"
        )
        assert scheduling.pods_per_cell() == 1, (
            f"pool {pool_id} spreads a cell over {scheduling.pods_per_cell()} pods, "
            f"so its workers do not share one host"
        )
        return naming.static_cell_addrs(
            spec=self._spec,
            release=self._release,
            cell_index=cell_index,
            worker_in_pod_index=worker_in_cell_index,
        )
