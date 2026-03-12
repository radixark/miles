"""Ray state actors for cross-process state sharing in testbed tests.

Provides lightweight Ray actors that hold mutable state accessible from
both the test driver and the controller/agent actors running in separate
processes.  Used by the MilesTestbed infrastructure.
"""

from __future__ import annotations

import ray

from miles.utils.ft.agents.types import MetricSample


@ray.remote(num_cpus=0, num_gpus=0)
class CollectorStateActor:
    """Shared metric state for RemoteControlledCollector.

    Tests call set_metrics() to inject hardware metrics (GPU unavailable,
    XID errors, NIC down, etc.) that node agents will scrape.
    """

    def __init__(self) -> None:
        self._metrics: list[MetricSample] = []

    def set_metrics(self, metrics: list[MetricSample]) -> None:
        self._metrics = metrics

    def get_metrics(self) -> list[MetricSample]:
        return self._metrics


@ray.remote(num_cpus=0, num_gpus=0)
class NotifierStateActor:
    """Stores notification records across Ray actor boundaries.

    The controller actor runs in a separate process, so a plain FakeNotifier's
    calls list is inaccessible from the test driver. This actor holds the
    records so both sides can access them via Ray RPCs.
    """

    def __init__(self) -> None:
        self._calls: list[tuple[str, str, str]] = []

    def record(self, title: str, content: str, severity: str) -> None:
        self._calls.append((title, content, severity))

    def get_calls(self) -> list[tuple[str, str, str]]:
        return list(self._calls)

    def clear(self) -> None:
        self._calls.clear()


@ray.remote(num_cpus=0, num_gpus=0)
class NodeManagerStateActor:
    """Holds node manager state across Ray actor boundaries.

    Stores mark/unmark state so both the controller actor and the test
    driver can access it via Ray RPCs.
    """

    def __init__(self) -> None:
        self._bad_nodes: set[str] = set()
        self._ever_marked_bad: set[str] = set()
        self._last_node_metadata: dict[str, str] | None = None

    def mark_bad(
        self,
        node_id: str,
        reason: str,
        node_metadata: dict[str, str] | None,
    ) -> None:
        self._bad_nodes.add(node_id)
        self._ever_marked_bad.add(node_id)
        self._last_node_metadata = node_metadata

    def clear_bad_nodes(self) -> None:
        self._bad_nodes.clear()

    def was_ever_marked_bad(self, node_id: str) -> bool:
        return node_id in self._ever_marked_bad
