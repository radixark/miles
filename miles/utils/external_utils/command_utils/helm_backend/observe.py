from __future__ import annotations

from collections.abc import Iterable

from miles.utils.pydantic_utils import FrozenStrictBaseModel

_MANY_PODS = 50
_MAX_LOG_REQUESTS = 100

RELEASE_LABEL = "app.kubernetes.io/instance"


class PodStatus(FrozenStrictBaseModel):
    name: str
    phase: str
    ready: bool
    restarts: int
    scheduling_gated: bool = False


class PodEvent(FrozenStrictBaseModel):
    pod_name: str
    reason: str
    message: str
    count: int = 1
    type: str = "Normal"


_REPORTED_EVENTS = 5


def startup_summary(pods: Iterable[PodStatus], events: Iterable[PodEvent] = ()) -> str:
    pods = list(pods)
    lines = [_pod_line(pods), *_event_lines(events)]
    return "\n".join(lines)


def _pod_line(pods: list[PodStatus]) -> str:
    if not pods:
        return "no pods yet"

    counts = {
        "running": sum(1 for pod in pods if pod.phase == "Running" and pod.ready),
        "starting": sum(1 for pod in pods if pod.phase == "Running" and not pod.ready),
        "pending": sum(1 for pod in pods if pod.phase == "Pending" and not pod.scheduling_gated),
        "gated": sum(1 for pod in pods if pod.scheduling_gated),
        "failed": sum(1 for pod in pods if pod.phase == "Failed"),
        "restarted": sum(1 for pod in pods if pod.restarts > 0),
    }
    reported = ", ".join(f"{count} {label}" for label, count in counts.items() if count)
    return f"{len(pods)} pods: {reported}"


def _event_lines(events: Iterable[PodEvent]) -> list[str]:
    warnings = [event for event in events if event.type == "Warning"]
    if not warnings:
        return []

    ranked = sorted(warnings, key=lambda event: (-event.count, event.pod_name, event.reason))
    lines = [
        f"  {event.pod_name}: {event.reason} x{event.count}: {event.message}" for event in ranked[:_REPORTED_EVENTS]
    ]
    if len(ranked) > _REPORTED_EVENTS:
        lines.append(f"  ... and {len(ranked) - _REPORTED_EVENTS} more warning events")
    return lines


def is_settled(pods: Iterable[PodStatus]) -> bool:
    pods = list(pods)
    return bool(pods) and all(pod.phase == "Failed" or (pod.phase == "Running" and pod.ready) for pod in pods)


def scale_hint(pods: Iterable[PodStatus]) -> str | None:
    count = len(list(pods))
    if count <= _MANY_PODS:
        return None
    return (
        f"this run has {count} pods; the cluster's own observability stack will show it better than "
        f"this summary does"
    )


def observability_boundary() -> str:
    return (
        "this launcher only prints pod phases, warning events and the orchestrator log; anything beyond that "
        "-- metrics, per-container history, logs of pods that already went away -- belongs to your cluster's "
        "own observability stack, which miles deliberately does not replace"
    )


def follow_log_command(*, namespace: str, workload: str, container: str = "orchestrator") -> list[str]:
    return ["kubectl", "logs", "--follow", "--namespace", namespace, f"statefulset/{workload}", "-c", container]


def release_log_command(*, namespace: str, release: str) -> list[str]:
    return [
        "kubectl",
        "logs",
        "--follow",
        "--namespace",
        namespace,
        "--selector",
        f"{RELEASE_LABEL}={release}",
        "--all-containers",
        "--prefix",
        "--max-log-requests",
        str(_MAX_LOG_REQUESTS),
    ]


def farewell(*, namespace: str, release: str, workload: str) -> str:
    return "\n".join(
        [
            "the run keeps going after this launcher exits",
            f"  orchestrator log: {' '.join(follow_log_command(namespace=namespace, workload=workload))}",
            f"  every pod of the run: {' '.join(release_log_command(namespace=namespace, release=release))}",
            f"  tear down: helm uninstall -n {namespace} {release}",
            f"  {observability_boundary()}",
        ]
    )
