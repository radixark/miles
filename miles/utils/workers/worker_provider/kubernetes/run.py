from __future__ import annotations

from collections.abc import Callable

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_provider.kubernetes.naming import static_worker_host
from miles.utils.workers.worker_provider.kubernetes.views.pod_info import CellLabelKeys
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    BaseWorkerSpec,
    HostAndPort,
    NamedHostAndPorts,
    ServeWorkerSpec,
    SpecMetaFn,
)


class PoolView(FrozenStrictBaseModel):
    ports: dict[str, int]
    worker_class: str | None
    meta: SpecMetaFn | None
    ranks_per_pod: int


class KubernetesRun(FrozenStrictBaseModel):
    namespace: str
    label_selector: str
    pools: dict[str, PoolView]
    kubernetes_client_factory: Callable[[], object]
    label_keys: CellLabelKeys


def kubernetes_run(
    *,
    specs: list[BaseWorkerSpec],
    namespace: str,
    label_selector: str,
    kubernetes_client_factory: Callable[[], object],
    num_gpus_per_node: int,
    label_keys: CellLabelKeys,
) -> KubernetesRun:
    return KubernetesRun(
        namespace=namespace,
        label_selector=label_selector,
        pool_ids={
            spec.name: _pool_view_of_spec(spec, num_gpus_per_node=num_gpus_per_node)
            for spec in specs
            if declares_dynamic_pool(spec)
        },
        kubernetes_client_factory=kubernetes_client_factory,
        label_keys=label_keys,
    )


def static_worker_provider(*, specs: list[BaseWorkerSpec], release: str) -> SimpleWorkerProvider:
    addrs: dict[str, NamedHostAndPorts] = {}
    cells: dict[str, list[str]] = {}
    pool_ids: dict[str, str] = {}
    worker_classes: dict[str, str] = {}

    for spec in specs:
        if declares_dynamic_pool(spec):
            continue
        if isinstance(spec, ServeWorkerSpec):
            worker_classes[spec.name] = spec.worker_class
        for cell_index in range(spec.scheduling.num_cells):
            cell_id = compute_cell_id(pool_id=spec.name, cell_index=cell_index)
            host = static_worker_host(release, spec.name, cell_index)
            worker_names = [
                compute_worker_name(cell_id=cell_id, worker_in_cell_index=index)
                for index in range(spec.scheduling.num_workers_per_cell)
            ]
            for worker_name in worker_names:
                addrs[worker_name] = {
                    port.name: HostAndPort(host=host, port=port.static_port) for port in spec.port_infos
                }
            cells[cell_id] = worker_names
            pool_ids[cell_id] = spec.name

    return SimpleWorkerProvider(addrs=addrs, cells=cells, pool_ids=pool_ids, worker_classes=worker_classes)


def declares_dynamic_pool(spec: BaseWorkerSpec) -> bool:
    return spec.scheduling.num_workers_per_cell * spec.scheduling.num_gpu_slots_per_worker > 0


def _pool_view_of_spec(spec: BaseWorkerSpec, *, num_gpus_per_node: int) -> PoolView:
    ranks_per_pod = _ranks_per_pod_of_spec(spec, num_gpus_per_node=num_gpus_per_node)
    return PoolView(
        ports={port.name: port.static_port for port in spec.port_infos},
        worker_class=spec.worker_class if isinstance(spec, ServeWorkerSpec) else None,
        meta=spec.meta,
        ranks_per_pod=ranks_per_pod,
    )


def _ranks_per_pod_of_spec(spec: BaseWorkerSpec, *, num_gpus_per_node: int) -> int:
    if not isinstance(spec, ServeWorkerSpec):
        return 1

    ranks_per_pod = min(spec.scheduling.num_workers_per_cell, num_gpus_per_node)
    _assert_rank_ports_are_free(spec, ranks_per_pod=ranks_per_pod)
    return ranks_per_pod


def _assert_rank_ports_are_free(spec: ServeWorkerSpec, *, ranks_per_pod: int) -> None:
    rpc_port = next(port.static_port for port in spec.port_infos if port.name == RPC_PORT_NAME)
    for port in spec.port_infos:
        if port.name == RPC_PORT_NAME:
            continue
        assert rpc_port + ranks_per_pod <= port.static_port or port.static_port + port.num_consecutive <= rpc_port, (
            f"spec '{spec.name}' serves {ranks_per_pod} ranks per pod from {RPC_PORT_NAME} port {rpc_port} "
            f"upwards, which reaches into the {port.num_consecutive} port(s) '{port.name}' claims from "
            f"{port.static_port}"
        )
