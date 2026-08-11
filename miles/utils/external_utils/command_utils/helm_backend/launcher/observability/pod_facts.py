from __future__ import annotations

from typing import NamedTuple

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.workers.k8s_types import Event, EventList, Pod, PodList

_SCHEDULING_GATED_REASON = "SchedulingGated"
_POD_WARNINGS_SELECTOR = "involvedObject.kind=Pod,type=Warning"


class ContainerKey(NamedTuple):
    pod: str
    container: str
    previous: bool


class ContainerRun(NamedTuple):
    key: ContainerKey
    container_id: str
    running: bool


def selected_pods(namespace: str, selector: str) -> list[Pod]:
    listed = Kubectl.get_json("pods", return_type=PodList, namespace=namespace, selector=selector)
    return listed.items if listed is not None else []


def pod_events(namespace: str, pods: list[Pod]) -> list[Event]:
    wanted = {pod.metadata.name for pod in pods}
    if not wanted:
        return []

    listed = Kubectl.get_json(
        "events", return_type=EventList, namespace=namespace, field_selector=_POD_WARNINGS_SELECTOR
    )
    events = listed.items if listed is not None else []
    return [event for event in events if event.involved_object.name in wanted]


def pod_phase(namespace: str, workload: str) -> str | None:
    described = Kubectl.get_json("pod", return_type=Pod, name=f"{workload}-0", namespace=namespace)
    if described is None:
        return None
    return phase_of_pod(described)


def status_changes(previous: list[Pod], current: list[Pod]) -> list[str]:
    before = {pod.metadata.name: pod for pod in previous}
    after = {pod.metadata.name: pod for pod in current}

    lines = [f"pod {name} is gone" for name in sorted(set(before) - set(after))]
    for name in sorted(after):
        described = _described(after[name])
        if (was := before.get(name)) is None:
            lines.append(f"pod {name} appeared: {described}")
        elif _described(was) != described:
            lines.append(f"pod {name} is now {described}")
    return lines


def container_runs(pods: list[Pod]) -> dict[ContainerKey, ContainerRun]:
    runs = [
        ContainerRun(
            key=ContainerKey(pod=pod.metadata.name, container=container.name, previous=previous),
            container_id=f"{pod.metadata.uid}/{container_id}",
            running=container.state.running is not None and not previous,
        )
        for pod in pods
        for container in pod.status.container_statuses
        for previous, container_id in (
            (False, container.container_id),
            (True, container.last_state.terminated.container_id if container.last_state.terminated else None),
        )
        if container_id
    ]
    return {run.key: run for run in runs}


def phase_of_pod(pod: Pod) -> str:
    return pod.status.phase or "Unknown"


def is_pod_ready(pod: Pod) -> bool:
    return any(condition.type == "Ready" and condition.status == "True" for condition in pod.status.conditions)


def restarts_of_pod(pod: Pod) -> int:
    return sum(container.restart_count for container in pod.status.container_statuses)


def is_pod_scheduling_gated(pod: Pod) -> bool:
    return any(
        condition.type == "PodScheduled" and condition.reason == _SCHEDULING_GATED_REASON
        for condition in pod.status.conditions
    )


def _described(pod: Pod) -> str:
    if is_pod_scheduling_gated(pod):
        return "Pending (scheduling gated)"
    phase = phase_of_pod(pod)
    if phase == "Running" and not is_pod_ready(pod):
        return "Running (not ready yet)"
    restarts = f", restarted {count} times" if (count := restarts_of_pod(pod)) else ""
    return f"{phase}{restarts}"
