from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import polling
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.pod_facts import (
    is_pod_ready,
    is_pod_scheduling_gated,
    phase_of_pod,
    pod_events,
    restarts_of_pod,
    selected_pods,
    status_changes,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.polling import polling_in_background
from miles.utils.workers.k8s_types import Event, Pod

logger = logging.getLogger(__name__)

_REPORTED_EVENTS = 5
_MANY_PODS = 100


@contextmanager
def with_cluster_info(*, namespace: str, selector: str) -> Iterator[None]:
    watcher = _ClusterInfoWatcher(namespace=namespace, selector=selector)
    with polling_in_background(
        watcher.report_changes,
        description="read the pods of this run",
        join_timeout=polling.POLL_INTERVAL_SECONDS,
    ):
        yield


class _ClusterInfoWatcher:
    def __init__(self, *, namespace: str, selector: str) -> None:
        self._namespace = namespace
        self._selector = selector
        self._pods: list[Pod] = []
        self._reported_events: set[tuple[str | None, str | None]] = set()
        self._reported_summary = ""
        self._hinted = False

    def report_changes(self) -> None:
        current = selected_pods(self._namespace, self._selector)
        for line in status_changes(self._pods, current):
            logger.info(line)
        self._pods = current

        if (summary := _pod_summary(current)) != self._reported_summary:
            logger.info(summary)
            self._reported_summary = summary
        if not self._hinted and (hint := _scale_hint(current)) is not None:
            logger.info(hint)
            self._hinted = True

        for line in _warning_lines(self._unreported_warnings()):
            logger.warning(line)

    def _unreported_warnings(self) -> list[Event]:
        events = pod_events(namespace=self._namespace, pods=self._pods)
        fresh = [event for event in events if _event_key(event) not in self._reported_events]
        self._reported_events.update(_event_key(event) for event in fresh)
        return fresh


def _pod_summary(pods: list[Pod]) -> str:
    if not pods:
        return "No pods yet"

    counts = {
        "running": sum(1 for pod in pods if phase_of_pod(pod) == "Running" and is_pod_ready(pod)),
        "starting": sum(1 for pod in pods if phase_of_pod(pod) == "Running" and not is_pod_ready(pod)),
        "pending": sum(1 for pod in pods if phase_of_pod(pod) == "Pending" and not is_pod_scheduling_gated(pod)),
        "gated": sum(1 for pod in pods if is_pod_scheduling_gated(pod)),
        "failed": sum(1 for pod in pods if phase_of_pod(pod) == "Failed"),
        "restarted": sum(1 for pod in pods if restarts_of_pod(pod) > 0),
    }
    reported = ", ".join(f"{count} {label}" for label, count in counts.items() if count)
    return f"{len(pods)} pods: {reported}"


def _scale_hint(pods: list[Pod]) -> str | None:
    if (count := len(pods)) <= _MANY_PODS:
        return None
    return (
        f"This run has {count} pods; the cluster's own observability stack will show it better than "
        f"this summary does"
    )


def _warning_lines(events: list[Event]) -> list[str]:
    ranked = sorted(
        (event for event in events if event.type == "Warning"),
        key=lambda event: (-event.count, event.involved_object.name or "", event.reason or ""),
    )
    lines = [
        f"{event.involved_object.name}: {event.reason} x{event.count}: {(event.message or '').strip()}"
        for event in ranked[:_REPORTED_EVENTS]
    ]
    if len(ranked) > _REPORTED_EVENTS:
        lines.append(f"... and {len(ranked) - _REPORTED_EVENTS} more warning events")
    return lines


def _event_key(event: Event) -> tuple[str | None, str | None]:
    return event.involved_object.name, event.reason
