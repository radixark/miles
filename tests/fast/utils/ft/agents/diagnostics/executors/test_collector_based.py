"""Tests for miles.utils.ft.agents.diagnostics.executors.collector_based."""

from __future__ import annotations

import asyncio

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.diagnostics.executors.collector_based import CollectorBasedNodeExecutor
from miles.utils.ft.agents.types import GaugeSample, MetricSample
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType


class _FakeCollector(BaseCollector):
    collect_interval: float = 10.0

    def __init__(self, metrics: list[MetricSample]) -> None:
        self._metrics = metrics

    def _collect_sync(self) -> list[MetricSample]:
        return self._metrics


class _CrashingCollector(BaseCollector):
    collect_interval: float = 10.0

    def _collect_sync(self) -> list[MetricSample]:
        raise RuntimeError("hw failure")


class _AlwaysPassDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        return Decision.no_fault("all checks passed")


class _AlwaysFailDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        return Decision(
            action=ActionType.ENTER_RECOVERY,
            reason="fault detected",
            trigger=TriggerType.HARDWARE,
            bad_node_ids=["node-0"],
        )


class TestCollectorBasedNodeExecutor:
    def test_pass_when_detector_returns_no_fault(self) -> None:
        collector = _FakeCollector(metrics=[GaugeSample(name="m", labels={}, value=1.0)])
        executor = CollectorBasedNodeExecutor(
            diagnostic_type="test",
            collector=collector,
            detector=_AlwaysPassDetector(),
        )

        result = asyncio.run(executor.run(node_id="node-0"))

        assert result.passed is True
        assert result.diagnostic_type == "test"

    def test_fail_when_detector_returns_fault(self) -> None:
        collector = _FakeCollector(metrics=[GaugeSample(name="m", labels={}, value=1.0)])
        executor = CollectorBasedNodeExecutor(
            diagnostic_type="test",
            collector=collector,
            detector=_AlwaysFailDetector(),
        )

        result = asyncio.run(executor.run(node_id="node-0"))

        assert result.passed is False
        assert "fault detected" in result.details

    def test_collector_exception_propagates(self) -> None:
        collector = _CrashingCollector()
        executor = CollectorBasedNodeExecutor(
            diagnostic_type="test",
            collector=collector,
            detector=_AlwaysPassDetector(),
        )

        try:
            asyncio.run(executor.run(node_id="node-0"))
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            assert "hw failure" in str(exc)

    def test_empty_metrics_pass(self) -> None:
        collector = _FakeCollector(metrics=[])
        executor = CollectorBasedNodeExecutor(
            diagnostic_type="test",
            collector=collector,
            detector=_AlwaysPassDetector(),
        )

        result = asyncio.run(executor.run(node_id="node-0"))

        assert result.passed is True
