from __future__ import annotations

import hashlib
from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.reconcile.k8s_types import Pod, PodStatus, parse_pod_object


class CellLabelKeys(FrozenStrictBaseModel):
    pool_id: str
    cell_ordinal: str
    pod_index: str
    cell_size: str
    meta_annotation_prefix: str
    gpu_ids_meta: str


class ParsedPod(FrozenStrictBaseModel):
    name: str
    cell_id: str
    cell_ordinal: int
    pool_id: str
    pod_index: int
    ready: bool
    pod_ip: str | None
    uid: str
    restart_count: int
    node_name: str | None
    meta: dict[str, str] = {}
    cell_size: int = 0
    subdomain: str | None = None
    gpu_ids: tuple[int, ...] = ()


def parse_pods(raw_pods: list[Any], *, keys: CellLabelKeys) -> list[ParsedPod]:
    pods = [parsed for pod in raw_pods if (parsed := parse_pod(pod, keys)) is not None]
    return sorted_by_pod_index(pods)


def parse_pod(raw_pod: Any, keys: CellLabelKeys) -> ParsedPod | None:
    pod = parse_pod_object(raw_pod)
    metadata = pod.metadata
    labels = metadata.labels
    pool_id = labels.get(keys.pool_id)
    cell_ordinal = labels.get(keys.cell_ordinal)
    if pool_id is None or cell_ordinal is None:
        return None

    status = pod.status
    meta = _read_meta(pod, keys)
    return ParsedPod(
        name=metadata.name,
        cell_id=compute_cell_id(pool_id=pool_id, cell_index=int(cell_ordinal)),
        cell_ordinal=int(cell_ordinal),
        pool_id=pool_id,
        pod_index=int(labels.get(keys.pod_index, 0)),
        ready=_is_ready(status),
        pod_ip=status.pod_ip,
        uid=metadata.uid,
        restart_count=sum(container.restart_count for container in status.container_statuses),
        node_name=pod.spec.node_name,
        meta=meta,
        cell_size=int(labels.get(keys.cell_size, 0)),
        subdomain=pod.spec.subdomain,
        gpu_ids=_parse_gpu_ids(meta.get(keys.gpu_ids_meta, "")),
    )


def cell_members_hash(pods: list[ParsedPod]) -> str:
    parts = [f"{pod.name}:{pod.uid}:{pod.restart_count}" for pod in sorted_by_pod_index(pods)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def sorted_by_pod_index(pods: list[ParsedPod]) -> list[ParsedPod]:
    return sorted(pods, key=lambda pod: pod.pod_index)


def _read_meta(pod: Pod, keys: CellLabelKeys) -> dict[str, str]:
    annotations = pod.metadata.annotations
    prefix = keys.meta_annotation_prefix
    return {key[len(prefix) :]: value for key, value in annotations.items() if key.startswith(prefix)}


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def _is_ready(status: PodStatus) -> bool:
    return any(condition.type == "Ready" and condition.status == "True" for condition in status.conditions)
