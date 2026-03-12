"""Sliding-window checker that warns when training nodes lack a registered node agent."""

from __future__ import annotations

import logging

from miles.utils.ft.utils.sliding_window import SlidingWindowCounter

logger = logging.getLogger(__name__)


class NodeAgentCoverageChecker:
    """Tracks node agent coverage over a sliding window and warns on sustained gaps.

    Each call to ``check()`` records whether each training node has a
    registered agent. Only after a node has been uncovered for
    *threshold* events within *window_seconds* does a warning fire,
    filtering out transient registration delays. Once warned, a node
    is not warned again until coverage is restored and then lost again.
    """

    def __init__(
        self,
        window_seconds: float = 300,
        threshold: int = 5,
    ) -> None:
        self._window_seconds = window_seconds
        self._threshold = threshold
        self._counters: dict[str, SlidingWindowCounter] = {}

    def check(
        self,
        subsystem_node_ids: set[str],
        registered_agent_node_ids: set[str],
    ) -> None:
        uncovered = subsystem_node_ids - registered_agent_node_ids
        covered = subsystem_node_ids & registered_agent_node_ids

        for node_id in covered:
            counter = self._counters.pop(node_id, None)
            if counter is not None and counter._notified:
                logger.info("Node agent coverage restored: %s", node_id)

        for node_id in uncovered:
            if node_id not in self._counters:
                self._counters[node_id] = SlidingWindowCounter(
                    window_seconds=self._window_seconds,
                    threshold=self._threshold,
                )
            counter = self._counters[node_id]
            counter.record(label=node_id)

            if counter.should_notify:
                logger.warning(
                    "Node %s has been running training without node agent: %s",
                    node_id,
                    counter.summary(),
                )
