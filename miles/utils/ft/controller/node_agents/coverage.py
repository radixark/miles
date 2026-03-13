"""Sliding-window checker that warns when subsystem nodes lack a registered node agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

from miles.utils.ft.utils.sliding_window import SlidingWindowCounter

logger = logging.getLogger(__name__)


class _SubsystemNodeKey(NamedTuple):
    subsystem_name: str
    node_id: str


@dataclass
class CoverageResult:
    """Structured output from a single coverage check call."""

    subsystem_name: str
    persistently_uncovered_node_ids: list[str] = field(default_factory=list)
    newly_restored_node_ids: list[str] = field(default_factory=list)


class NodeAgentCoverageChecker:
    """Tracks node agent coverage over a sliding window and warns on sustained gaps.

    Each call to ``check()`` records whether each subsystem node has a
    registered agent. Counters are keyed by ``_SubsystemNodeKey``
    so the same physical node tracked in different subsystems has independent
    coverage state.

    Only after a node has been uncovered for *threshold* events within
    *window_seconds* does it appear in the returned
    ``CoverageResult.persistently_uncovered_node_ids``, filtering out
    transient registration delays. Once reported, a node is not reported
    again until coverage is restored and then lost again.
    """

    def __init__(
        self,
        window_seconds: float = 300,
        threshold: int = 5,
    ) -> None:
        self._window_seconds = window_seconds
        self._threshold = threshold
        self._counters: dict[_SubsystemNodeKey, SlidingWindowCounter] = {}

    def check(
        self,
        subsystem_name: str,
        subsystem_node_ids: set[str],
        registered_agent_node_ids: set[str],
    ) -> CoverageResult:
        result = CoverageResult(subsystem_name=subsystem_name)
        uncovered = subsystem_node_ids - registered_agent_node_ids
        covered = subsystem_node_ids & registered_agent_node_ids

        for node_id in covered:
            key = _SubsystemNodeKey(subsystem_name=subsystem_name, node_id=node_id)
            counter = self._counters.pop(key, None)
            if counter is not None and counter._notified:
                logger.info("Node agent coverage restored: %s/%s", subsystem_name, node_id)
                result.newly_restored_node_ids.append(node_id)

        for node_id in uncovered:
            key = _SubsystemNodeKey(subsystem_name=subsystem_name, node_id=node_id)
            if key not in self._counters:
                self._counters[key] = SlidingWindowCounter(
                    window_seconds=self._window_seconds,
                    threshold=self._threshold,
                )
            counter = self._counters[key]
            counter.record(label=f"{subsystem_name}/{node_id}")

            if counter.should_notify:
                logger.warning(
                    "Node %s has been running %s without node agent: %s",
                    node_id,
                    subsystem_name,
                    counter.summary(),
                )
                result.persistently_uncovered_node_ids.append(node_id)

        return result
