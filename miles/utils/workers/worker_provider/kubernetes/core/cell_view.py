from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from miles.utils.http_utils import wrap_ipv6
from miles.utils.misc import merge_asserting_consistency
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.connection_config import WorkerPodMetadata
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_provider.kubernetes.core import pod_view
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

if TYPE_CHECKING:
    from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesRunInfo


class KubernetesWorkerInfo(FrozenStrictBaseModel):
    pod: pod_view.ParsedPod
    name: str
    worker_in_pod_index: int
    gpu_ids: list[int]


class PodIdentity(FrozenStrictBaseModel):
    name: str
    uid: str


class CellIncarnation(FrozenStrictBaseModel):
    cell_id: str
    workers_hash: str
    pods: list[PodIdentity]


def compute_cell_info(cell_id: str, *, pods: list[pod_view.ParsedPod], run: KubernetesRunInfo) -> CellInfo | None:
    if not pods:
        return None

    pool_id = pods[0].pool_id
    meta = _metadata_of_cell(pods).static_meta.resolve(cell_index=pods[0].cell_index) | _pod_meta_of_cell(pods)

    return CellInfo(
        cell_id=cell_id,
        pool_id=pool_id,
        alive=all(pod.ready and not pod.deleting for pod in pods) and _has_all_pods(pods),
        worker_names=[worker.name for worker in workers_of_pods(pods, run=run)],
        workers_hash=pod_view.cell_members_hash(pods),
        meta=meta,
    )


def compute_debug_cell_incarnation(cell_id: str, *, pods: list[pod_view.ParsedPod]) -> CellIncarnation | None:
    if not pods:
        return None

    return CellIncarnation(
        cell_id=cell_id,
        workers_hash=pod_view.cell_members_hash(pods),
        pods=[PodIdentity(name=pod.name, uid=pod.uid) for pod in pod_view.sorted_by_pod_in_cell_index(pods)],
    )


def compute_worker_infos(cell_id: str, *, pods: list[pod_view.ParsedPod], run: KubernetesRunInfo) -> list[WorkerInfo]:
    assert pods, f"cell {cell_id} has no observed worker pods, so it cannot be driven"
    indices = [pod.pod_in_cell_index for pod in pods]
    assert indices == list(range(len(pods))) and _has_all_pods(
        pods
    ), f"cell {cell_id} is missing pods: observed {indices} of {max(pod.cell_size for pod in pods)}"

    return [_compute_worker_info(worker, run=run) for worker in workers_of_pods(pods, run=run)]


def workers_of_pods(pods: list[pod_view.ParsedPod], *, run: KubernetesRunInfo) -> list[KubernetesWorkerInfo]:
    if not pods:
        return []
    metadata = _metadata_of_cell(pods)
    return [worker for pod in pods for worker in _workers_of_pod(pod, metadata=metadata)]


def addrs_of_worker(worker: KubernetesWorkerInfo, *, run: KubernetesRunInfo) -> NamedHostAndPorts:
    host = _host_of_pod(worker.pod, namespace=run.namespace)
    ports = worker.pod.worker_metadata.port_infos
    assert ports, f"spec {worker.pod.pool_id} declares no ports, so {worker.name} has no address"
    return {
        port.name: HostAndPort(
            host=host,
            port=port.effective_static_port(worker_in_pod_index=worker.worker_in_pod_index),
        )
        for port in ports
    }


def _compute_worker_info(worker: KubernetesWorkerInfo, *, run: KubernetesRunInfo) -> WorkerInfo:
    return WorkerInfo(
        name=worker.name,
        generation=worker.pod.restart_count,
        self_addrs=addrs_of_worker(worker, run=run),
        gpu_ids=list(worker.gpu_ids),
        worker_class=worker.pod.worker_metadata.worker_class,
    )


def _workers_of_pod(pod: pod_view.ParsedPod, *, metadata: WorkerPodMetadata) -> list[KubernetesWorkerInfo]:
    workers_per_pod = metadata.workers_per_pod
    assert len(pod.gpu_ids) % workers_per_pod == 0, (
        f"pod {pod.name} was annotated with {len(pod.gpu_ids)} gpus for the {workers_per_pod} workers it serves, "
        f"so no worker owns an equal share of them"
    )
    gpus_per_worker = len(pod.gpu_ids) // workers_per_pod
    return [
        KubernetesWorkerInfo(
            pod=pod,
            name=compute_worker_name(
                pool_id=pod.pool_id,
                cell_index=pod.cell_index,
                worker_in_cell_index=pod.pod_in_cell_index * workers_per_pod + worker_in_pod_index,
            ),
            worker_in_pod_index=worker_in_pod_index,
            gpu_ids=list(
                pod.gpu_ids[worker_in_pod_index * gpus_per_worker : (worker_in_pod_index + 1) * gpus_per_worker]
            ),
        )
        for worker_in_pod_index in range(workers_per_pod)
    ]


def _host_of_pod(pod: pod_view.ParsedPod, *, namespace: str) -> str:
    if pod.pod_ip:
        return wrap_ipv6(pod.pod_ip)
    assert pod.subdomain, f"worker {pod.name} has neither a pod ip nor a headless service"
    return f"{pod.name}.{pod.subdomain}.{namespace}.svc"


def _has_all_pods(pods: list[pod_view.ParsedPod]) -> bool:
    expected = max(pod.cell_size for pod in pods)
    if not expected:
        return True
    return sorted(pod.pod_in_cell_index for pod in pods) == list(range(expected))


def _metadata_of_cell(pods: list[pod_view.ParsedPod]) -> WorkerPodMetadata:
    metadata = pods[0].worker_metadata
    assert all(
        pod.worker_metadata == metadata for pod in pods
    ), f"cell {pods[0].cell_id} has inconsistent worker metadata across its pods"
    return metadata


def _pod_meta_of_cell(pods: list[pod_view.ParsedPod]) -> dict[str, str]:
    return functools.reduce(merge_asserting_consistency, (pod.meta for pod in pods), {})
