"""Tests for miles.utils.ft.agents.diagnostics.executors.collector_based.

The CollectorBasedNodeExecutor used to directly import and instantiate controller-layer
classes (BaseFaultDetector, InMemoryMetricStore, MiniWandb, DetectorContext). It now
takes a SampleEvaluator callable, keeping the agents layer free of controller imports.
"""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Sequence
from pathlib import Path

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.diagnostics.executors.collector_based import CollectorBasedNodeExecutor
from miles.utils.ft.agents.types import CounterSample, GaugeSample, MetricSample


class _FakeCollector(BaseCollector):
    collect_interval: float = 10.0

    def __init__(self, metrics: list[MetricSample]) -> None:
        self._metrics = metrics

    def _collect_sync(self) -> list[MetricSample]:
        return self._metrics


class _SlowCollector(BaseCollector):
    collect_interval: float = 999.0

    def _collect_sync(self) -> list[MetricSample]:
        import time

        time.sleep(10)
        return []


class _CrashingCollector(BaseCollector):
    collect_interval: float = 10.0

    def _collect_sync(self) -> list[MetricSample]:
        raise RuntimeError("hw failure")


def _always_pass_evaluator(
    node_id: str,
    samples: Sequence[GaugeSample | CounterSample],
) -> tuple[bool, str]:
    return True, "all checks passed"


def _always_fail_evaluator(
    node_id: str,
    samples: Sequence[GaugeSample | CounterSample],
) -> tuple[bool, str]:
    return False, "fault detected"


class TestCollectorBasedNodeExecutor:
    def test_pass_when_evaluator_returns_true(self) -> None:
        collector = _FakeCollector(metrics=[GaugeSample(name="m", labels={}, value=1.0)])
        executor = CollectorBasedNodeExecutor(
            diagnostic_type="test",
            collector=collector,
            evaluator=_always_pass_evaluator,
        )

        result = asyncio.run(executor.run(node_id="node-0"))

        assert result.passed is True
        assert result.diagnostic_type == "test"

    def test_fail_when_evaluator_returns_false(self) -> None:
        collector = _FakeCollector(metrics=[GaugeSample(name="m", labels={}, value=1.0)])
        executor = CollectorBasedNodeExecutor(
            diagnostic_type="test",
            collector=collector,
            evaluator=_always_fail_evaluator,
        )

        result = asyncio.run(executor.run(node_id="node-0"))

        assert result.passed is False
        assert "fault detected" in result.details

    def test_collector_exception_propagates(self) -> None:
        collector = _CrashingCollector()
        executor = CollectorBasedNodeExecutor(
            diagnostic_type="test",
            collector=collector,
            evaluator=_always_pass_evaluator,
        )

        with pytest.raises(RuntimeError, match="hw failure"):
            asyncio.run(executor.run(node_id="node-0"))

    def test_empty_metrics_pass(self) -> None:
        collector = _FakeCollector(metrics=[])
        executor = CollectorBasedNodeExecutor(
            diagnostic_type="test",
            collector=collector,
            evaluator=_always_pass_evaluator,
        )

        result = asyncio.run(executor.run(node_id="node-0"))

        assert result.passed is True


class TestCollectorBasedTimeout:
    def test_caller_timeout_is_respected(self) -> None:
        """Previously timeout_seconds was accepted but ignored — the collector
        used its own internal timeout (collect_interval * 2). Now the caller's
        timeout_seconds is enforced via asyncio.wait_for."""
        collector = _SlowCollector()
        executor = CollectorBasedNodeExecutor(
            diagnostic_type="test",
            collector=collector,
            evaluator=_always_pass_evaluator,
        )

        result = asyncio.run(executor.run(node_id="node-0", timeout_seconds=1))

        assert result.passed is False
        assert "timed out" in result.details


class TestCollectorBasedHasNoControllerImports:
    """The module used to directly import controller-layer classes.
    Now it only depends on agents-layer and utils-layer types.
    """

    def test_no_controller_imports_in_collector_based(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[7]
            / "miles"
            / "utils"
            / "ft"
            / "agents"
            / "diagnostics"
            / "executors"
            / "collector_based.py"
        )
        source = source_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert (
                        "controller" not in alias.name
                    ), f"collector_based.py still imports from controller: {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert (
                    "controller" not in node.module
                ), f"collector_based.py still imports from controller: {node.module}"
