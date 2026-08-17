from __future__ import annotations

from collections.abc import Iterable

from miles.utils.function_registry import load_function
from miles.utils.http_utils import wait_tcp_ready, wrap_ipv6
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm import naming
from miles.utils.workers.worker_provider.utils import build_rpc_handle
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    BaseWorkerSpec,
    HostAndPort,
    NamedHostAndPorts,
    ServeWorkerSpec,
)

_STATIC_ADDRS_READY_TIMEOUT_SECONDS = 600.0


class StaticWorkerProvider(BaseWorkerProvider):
    def __init__(
        self, *, pool_id: str, addrs_by_worker: dict[str, NamedHostAndPorts], worker_class: str | None
    ) -> None:
        assert addrs_by_worker, f"pool {pool_id} is addressed statically, so it needs at least one cell"
        self._pool_id = pool_id
        self._addrs_by_worker = addrs_by_worker
        self._worker_class = worker_class

    @classmethod
    def of_release(cls, *, release: str, spec: BaseWorkerSpec) -> StaticWorkerProvider:
        scheduling = spec.scheduling
        assert scheduling.pods_per_cell() == 1, (
            f"pool {spec.name} spreads a cell over {scheduling.pods_per_cell()} pods, "
            f"so its workers do not share one host"
        )
        return cls(
            pool_id=spec.name,
            addrs_by_worker={
                compute_worker_name(
                    pool_id=spec.name, cell_index=cell_index, worker_in_cell_index=worker_in_cell_index
                ): naming.static_cell_addrs(
                    spec=spec,
                    release=release,
                    cell_index=cell_index,
                    worker_in_pod_index=worker_in_cell_index,
                )
                for cell_index in range(scheduling.num_cells)
                for worker_in_cell_index in range(scheduling.num_workers_per_cell)
            },
            worker_class=spec.worker_class if isinstance(spec, ServeWorkerSpec) else None,
        )

    @classmethod
    def of_rpc_addrs(cls, *, pool_id: str, addrs: list[HostAndPort], worker_class: str) -> StaticWorkerProvider:
        return cls(
            pool_id=pool_id,
            addrs_by_worker={
                compute_worker_name(pool_id=pool_id, cell_index=cell_index): {RPC_PORT_NAME: addr}
                for cell_index, addr in enumerate(addrs)
            },
            worker_class=worker_class,
        )

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return self._addrs_of(worker_name)

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        raise NotImplementedError(f"{type(self).__name__} answers addresses, it does not enumerate workers")

    def get_handle(self, worker_name: str) -> BaseWorkerHandle:
        assert (
            self._worker_class is not None
        ), f"pool {self._pool_id} is launched as a command rather than served, so its rpc methods are unknown"
        return build_rpc_handle(worker_class=load_function(self._worker_class), addrs=self._addrs_of(worker_name))

    def _addrs_of(self, worker_name: str) -> NamedHostAndPorts:
        addrs = self._addrs_by_worker.get(worker_name)
        assert addrs is not None, (
            f"{worker_name} is not a worker of pool {self._pool_id}, which addresses "
            f"{sorted(self._addrs_by_worker)} statically"
        )
        return addrs


def wait_static_addrs_ready(addrs: Iterable[HostAndPort]) -> None:
    for addr in addrs:
        wait_tcp_ready(addr.host, addr.port, timeout=_STATIC_ADDRS_READY_TIMEOUT_SECONDS)


def parse_host_and_port(addr: str) -> HostAndPort:
    host, separator, port = addr.rpartition(":")
    assert separator and port.isdigit() and "/" not in addr, f"static address {addr!r} must be host:port"
    assert host, f"static address {addr!r} names a port but no host to dial it on"
    assert ":" not in host or (host.startswith("[") and host.endswith("]")), (
        f"static address {addr!r} reads as a bare ipv6 address, whose own colons cannot be told from the port "
        f"separator; write it as [{host}]:<port>"
    )
    return HostAndPort(host=wrap_ipv6(host), port=int(port))
