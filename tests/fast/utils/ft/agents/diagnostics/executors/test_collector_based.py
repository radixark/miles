"""Tests for miles.utils.ft.agents.diagnostics.executors.collector_based."""

from __future__ import annotations

import asyncio

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.diagnostics.executors.collector_based import CollectorBasedNodeExecutor
from miles.utils.ft.agents.types import GaugeSample, MetricSample
from miles.utils.ft.controller.types import MetricQueryProtocol, NodeFault


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


def _check_always_pass(store: MetricQueryProtocol) -> list[NodeFault]:
    return []


def _check_always_fail(store: MetricQueryProtocol) -> list[NodeFault]:
    return [NodeFault(node_id="node-0", reason="fault detected")]


class TestCollectorBasedNodeExecutor:
    def test_pass_when_check_fn_returns_no_faults(self) -> None:
        collector = _FakeCollector(
            metrics=[GaugeSample(name="m", labels={}, value=1.0)]
        )
        executor = CollectorBasedNodeExecutor(
            diagnostic_type="test",
            collector=collector,
            check_fn=_check_always_pass,
        )

        result = asyncio.run(executor.run(node_id="node-0"))

        assert result.passed is True
        assert result.diagnostic_type == "test"

    def test_fail_when_check_fn_returns_faults(self) -> None:
        collector = _FakeCollector(
            metrics=[GaugeSample(name="m", labels={}, value=1.0)]
        )
        executor = CollectorBasedNodeExecutor(
            diagnostic_type="test",
            collector=collector,
            check_fn=_check_always_fail,
        )

        result = asyncio.run(executor.run(node_id="node-0"))

        assert result.passed is False
        assert "fault detected" in result.details

    def test_collector_exception_propagates(self) -> None:
        collector = _CrashingCollector()
        executor = CollectorBasedNodeExecutor(
            diagnostic_type="test",
            collector=collector,
            check_fn=_check_always_pass,
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
            check_fn=_check_always_pass,
        )

        result = asyncio.run(executor.run(node_id="node-0"))

        assert result.passed is True
