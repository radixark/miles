from __future__ import annotations

import hashlib

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.k8s_types import Pod, PodStatus
from miles.utils.workers.naming import compute_cell_id


class CellLabelKeys(FrozenStrictBaseModel):
    pool_id: str
    cell_index: str
    pod_in_cell_index: str
    cell_size_annotation: str
    meta_annotation_prefix: str
    gpu_ids_meta: str
    base_gpu_id_annotation: str


class ParsedPod(FrozenStrictBaseModel):
    name: str
    cell_id: str
    cell_index: int
    pool_id: str
    pod_in_cell_index: int
    ready: bool
    deleting: bool
    pod_ip: str | None
    uid: str
    restart_count: int
    meta: dict[str, str]
    cell_size: int
    subdomain: str | None
    gpu_ids: tuple[int, ...]


def parse_pods(pods: list[Pod], *, keys: CellLabelKeys) -> list[ParsedPod]:
    parsed_pods = [parsed for pod in pods if (parsed := parse_pod(pod, keys)) is not None]
    return sorted_by_pod_in_cell_index(parsed_pods)


def parse_pod(pod: Pod, keys: CellLabelKeys) -> ParsedPod | None:
    metadata = pod.metadata
    labels = metadata.labels
    pool_id = labels.get(keys.pool_id)
    if pool_id is None or keys.cell_index not in labels:
        return None

    status = pod.status
    meta = _read_meta(pod, keys)
    cell_index = int(labels[keys.cell_index])
    return ParsedPod(
        name=metadata.name,
        cell_id=compute_cell_id(pool_id=pool_id, cell_index=cell_index),
        cell_index=cell_index,
        pool_id=pool_id,
        pod_in_cell_index=int(labels.get(keys.pod_in_cell_index, 0)),
        ready=_is_ready(status),
        deleting=metadata.deletion_timestamp is not None,
        pod_ip=status.pod_ip,
        uid=metadata.uid,
        restart_count=sum(container.restart_count for container in status.container_statuses),
        meta=meta,
        cell_size=int(metadata.annotations.get(keys.cell_size_annotation, 0)),
        subdomain=pod.spec.subdomain,
        gpu_ids=_parse_gpu_ids(meta.get(keys.gpu_ids_meta, "")),
    )


def cell_members_hash(pods: list[ParsedPod]) -> str:
    parts = [f"{pod.name}:{pod.uid}:{pod.restart_count}:{pod.deleting}" for pod in sorted_by_pod_in_cell_index(pods)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def sorted_by_pod_in_cell_index(pods: list[ParsedPod]) -> list[ParsedPod]:
    return sorted(pods, key=lambda pod: pod.pod_in_cell_index)


def _read_meta(pod: Pod, keys: CellLabelKeys) -> dict[str, str]:
    annotations = pod.metadata.annotations
    prefix = keys.meta_annotation_prefix
    return {key[len(prefix) :]: value for key, value in annotations.items() if key.startswith(prefix)}


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def _is_ready(status: PodStatus) -> bool:
    return any(condition.type == "Ready" and condition.status == "True" for condition in status.conditions)
