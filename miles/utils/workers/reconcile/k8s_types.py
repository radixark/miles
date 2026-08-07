# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class PodMetadata(FrozenStrictBaseModel):
    name: str
    uid: str
    labels: dict[str, str] = {}
    annotations: dict[str, str] = {}


class PodSpec(FrozenStrictBaseModel):
    node_name: str | None = None
    subdomain: str | None = None


class PodCondition(FrozenStrictBaseModel):
    type: str
    status: str


class ContainerStatus(FrozenStrictBaseModel):
    restart_count: int


class PodStatus(FrozenStrictBaseModel):
    pod_ip: str | None = None
    conditions: list[PodCondition] = []
    container_statuses: list[ContainerStatus] = []


class Pod(FrozenStrictBaseModel):
    metadata: PodMetadata
    spec: PodSpec
    status: PodStatus


def parse_pod_object(obj: Any) -> Pod:
    metadata = _field(obj, "metadata")
    spec = _field(obj, "spec")
    status = _field(obj, "status")

    return Pod(
        metadata=PodMetadata(
            name=_field(metadata, "name"),
            uid=_field(metadata, "uid"),
            labels=_field(metadata, "labels") or {},
            annotations=_field(metadata, "annotations") or {},
        ),
        spec=PodSpec(node_name=_field(spec, "node_name"), subdomain=_field(spec, "subdomain")),
        status=PodStatus(
            pod_ip=_field(status, "pod_ip"),
            conditions=[
                PodCondition(type=_field(condition, "type"), status=_field(condition, "status"))
                for condition in _field(status, "conditions") or []
            ],
            container_statuses=[
                ContainerStatus(restart_count=_field(container_status, "restart_count"))
                for container_status in _field(status, "container_statuses") or []
            ],
        ),
    )


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
