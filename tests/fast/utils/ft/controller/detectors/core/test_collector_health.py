"""Tests for CollectorHealthDetector.

Previously there was no mechanism to detect when a node agent's collectors
were persistently failing, causing the controller to operate blind on those
nodes — relying on stale or absent metrics without any warning.
"""
from __future__ import annotations

from tests.fast.utils.ft.utils import make_detector_context, make_fake_metric_store

from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.detectors.core.collector_health import CollectorHealthDetector
from miles.utils.ft.controller.types import ActionType, TriggerType


def _inject_collector_failures(
    store: object,
    *,
    node_id: str,
    collector: str,
    failures: int,
) -> None:
    store.ingest_samples(  # type: ignore[attr-defined]
        target_id=node_id,
        samples=[
            GaugeSample(
                name="ft_collector_consecutive_failures",
                labels={"collector": collector, "node_id": node_id},
                value=float(failures),
            ),
        ],
    )


class TestCollectorHealthDetector:
    def test_no_data_returns_no_fault(self) -> None:
        store = make_fake_metric_store()
        detector = CollectorHealthDetector()
        decision = detector.evaluate(make_detector_context(metric_store=store))
        assert decision.action == ActionType.NONE

    def test_healthy_collectors_returns_no_fault(self) -> None:
        store = make_fake_metric_store()
        _inject_collector_failures(store, node_id="node-0", collector="GpuCollector", failures=0)
        detector = CollectorHealthDetector()
        decision = detector.evaluate(make_detector_context(
            metric_store=store,
            active_node_ids={"node-0"},
        ))
        assert decision.action == ActionType.NONE

    def test_critical_collector_over_threshold_triggers_notify(self) -> None:
        store = make_fake_metric_store()
        _inject_collector_failures(store, node_id="node-0", collector="GpuCollector", failures=15)
        detector = CollectorHealthDetector(failure_threshold=10)
        decision = detector.evaluate(make_detector_context(
            metric_store=store,
            active_node_ids={"node-0"},
        ))
        assert decision.action == ActionType.NOTIFY_HUMAN
        assert decision.trigger == TriggerType.TELEMETRY_BLIND
        assert "node-0" in decision.reason

    def test_non_critical_collector_ignored(self) -> None:
        store = make_fake_metric_store()
        _inject_collector_failures(store, node_id="node-0", collector="KmsgCollector", failures=100)
        detector = CollectorHealthDetector(failure_threshold=10)
        decision = detector.evaluate(make_detector_context(
            metric_store=store,
            active_node_ids={"node-0"},
        ))
        assert decision.action == ActionType.NONE

    def test_inactive_node_ignored(self) -> None:
        store = make_fake_metric_store()
        _inject_collector_failures(store, node_id="node-1", collector="GpuCollector", failures=20)
        detector = CollectorHealthDetector(failure_threshold=10)
        decision = detector.evaluate(make_detector_context(
            metric_store=store,
            active_node_ids={"node-0"},
        ))
        assert decision.action == ActionType.NONE

    def test_below_threshold_no_fault(self) -> None:
        store = make_fake_metric_store()
        _inject_collector_failures(store, node_id="node-0", collector="GpuCollector", failures=9)
        detector = CollectorHealthDetector(failure_threshold=10)
        decision = detector.evaluate(make_detector_context(
            metric_store=store,
            active_node_ids={"node-0"},
        ))
        assert decision.action == ActionType.NONE
