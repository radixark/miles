from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from miles.utils.external_utils.command_utils.helm_backend.observe import PodEvent, PodStatus

_SCHEDULING_GATED_REASON = "SchedulingGated"


def release_pods(namespace: str, release: str) -> list[PodStatus]:
    pods = (
        _core_api()
        .list_namespaced_pod(namespace=namespace, label_selector=f"app.kubernetes.io/instance={release}")
        .items
    )
    return [_pod_status(pod) for pod in pods]


def pod_events(namespace: str, pods: Iterable[PodStatus]) -> list[PodEvent]:
    wanted = {pod.name for pod in pods}
    if not wanted:
        return []
    events = _core_api().list_namespaced_event(namespace=namespace, field_selector="involvedObject.kind=Pod").items
    return [_pod_event(event) for event in events if event.involved_object.name in wanted]


def pod_phase(namespace: str, workload: str) -> str | None:
    from kubernetes import client

    try:
        return _core_api().read_namespaced_pod(name=f"{workload}-0", namespace=namespace).status.phase
    except client.ApiException as exception:
        if exception.status == 404:
            return None
        raise


def _pod_event(event) -> PodEvent:
    return PodEvent(
        pod_name=event.involved_object.name,
        reason=event.reason or "Unknown",
        message=(event.message or "").strip(),
        count=event.count or 1,
        type=event.type or "Normal",
    )


def _pod_status(pod) -> PodStatus:
    status = pod.status
    conditions = status.conditions or []
    return PodStatus(
        name=pod.metadata.name,
        phase=status.phase or "Unknown",
        ready=any(condition.type == "Ready" and condition.status == "True" for condition in conditions),
        restarts=sum(container.restart_count for container in (status.container_statuses or [])),
        scheduling_gated=any(
            condition.type == "PodScheduled" and condition.reason == _SCHEDULING_GATED_REASON
            for condition in conditions
        ),
    )


@lru_cache(maxsize=1)
def _core_api():
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.CoreV1Api()
