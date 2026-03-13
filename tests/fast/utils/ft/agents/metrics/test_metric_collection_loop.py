"""Tests for MetricCollectionLoop — start/stop lifecycle and collector error isolation."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from tests.fast.utils.ft.utils import FailingCloseCollector, FailingCollector

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.metrics.metric_collection_loop import MetricCollectionLoop, _inject_node_id
from miles.utils.ft.agents.types import CounterSample, GaugeSample


class _PassCollector(BaseCollector):
    """Collector that returns a fixed metric list."""

    collect_interval: float = 0.01

    def __init__(self, metrics: list[GaugeSample] | None = None) -> None:
        self._metrics = metrics or []
        self.collect_count = 0

    def _collect_sync(self) -> list[GaugeSample]:
        self.collect_count += 1
        return list(self._metrics)


def _make_exporter() -> MagicMock:
    return MagicMock()


# ===================================================================
# start
# ===================================================================


class TestStart:
    @pytest.mark.anyio
    async def test_creates_tasks_for_each_collector(self) -> None:
        c1 = _PassCollector()
        c2 = _PassCollector()
        exporter = _make_exporter()
        loop = MetricCollectionLoop(node_id="n0", collectors=[c1, c2], exporter=exporter)

        await loop.start()

        assert len(loop.tasks) == 2
        await loop.stop()

    @pytest.mark.anyio
    async def test_start_after_stop_is_noop(self) -> None:
        collector = _PassCollector()
        exporter = _make_exporter()
        loop = MetricCollectionLoop(node_id="n0", collectors=[collector], exporter=exporter)

        await loop.start()
        await loop.stop()
        await loop.start()

        assert len(loop.tasks) == 0

    @pytest.mark.anyio
    async def test_start_twice_is_idempotent(self) -> None:
        collector = _PassCollector()
        exporter = _make_exporter()
        loop = MetricCollectionLoop(node_id="n0", collectors=[collector], exporter=exporter)

        await loop.start()
        await loop.start()

        assert len(loop.tasks) == 1
        await loop.stop()


# ===================================================================
# stop
# ===================================================================


class TestStop:
    @pytest.mark.anyio
    async def test_stop_cancels_tasks_and_closes_collectors(self) -> None:
        collector = _PassCollector()
        exporter = _make_exporter()
        loop = MetricCollectionLoop(node_id="n0", collectors=[collector], exporter=exporter)

        await loop.start()
        assert len(loop.tasks) == 1
        await loop.stop()

        assert len(loop.tasks) == 0

    @pytest.mark.anyio
    async def test_stop_is_idempotent(self) -> None:
        collector = _PassCollector()
        exporter = _make_exporter()
        loop = MetricCollectionLoop(node_id="n0", collectors=[collector], exporter=exporter)

        await loop.start()
        await loop.stop()
        await loop.stop()

    @pytest.mark.anyio
    async def test_close_exception_is_isolated(self) -> None:
        """One collector's close() failure must not prevent other collectors from closing."""
        failing = FailingCloseCollector(collect_interval=0.01)
        normal = _PassCollector()
        exporter = _make_exporter()
        loop = MetricCollectionLoop(
            node_id="n0",
            collectors=[failing, normal],
            exporter=exporter,
        )

        await loop.start()
        await loop.stop()

        assert failing.close_called


# ===================================================================
# _run_single_collector — collection + error isolation
# ===================================================================


class TestRunSingleCollector:
    @pytest.mark.anyio
    async def test_successful_collect_updates_exporter(self) -> None:
        sample = GaugeSample(name="m", labels={}, value=1.0)
        collector = _PassCollector(metrics=[sample])
        exporter = _make_exporter()
        loop = MetricCollectionLoop(node_id="n0", collectors=[collector], exporter=exporter)

        await loop.start()
        await asyncio.sleep(0.05)
        await loop.stop()

        assert collector.collect_count >= 1
        exporter.update_metrics.assert_called()
        call_args = exporter.update_metrics.call_args_list[0]
        assert len(call_args[0][0]) == 1
        assert call_args[0][0][0].name == "m"

    @pytest.mark.anyio
    async def test_collector_samples_get_node_id_injected(self) -> None:
        """Node agent collectors don't include node_id in their labels.
        Without injection, Prometheus backend would lack node_id on hardware
        metrics, causing detectors to crash or silently fail."""
        sample = GaugeSample(name="gpu_temp", labels={"gpu": "0"}, value=65.0)
        collector = _PassCollector(metrics=[sample])
        exporter = _make_exporter()
        loop = MetricCollectionLoop(node_id="my-node", collectors=[collector], exporter=exporter)

        await loop.start()
        await asyncio.sleep(0.05)
        await loop.stop()

        assert exporter.update_metrics.call_count >= 1
        first_call_samples = exporter.update_metrics.call_args_list[0][0][0]
        metric_sample = first_call_samples[0]
        assert metric_sample.labels["node_id"] == "my-node"
        assert metric_sample.labels["gpu"] == "0"

    @pytest.mark.anyio
    async def test_collect_exception_does_not_kill_loop(self) -> None:
        """A collector that raises on collect() should not stop the loop."""
        collector = FailingCollector(collect_interval=0.01)
        exporter = _make_exporter()
        loop = MetricCollectionLoop(node_id="n0", collectors=[collector], exporter=exporter)

        await loop.start()
        await asyncio.sleep(0.05)
        await loop.stop()

        assert collector.call_count >= 2
        staleness_names = {"ft_collector_consecutive_failures", "ft_collector_last_success_timestamp"}
        for call in exporter.update_metrics.call_args_list:
            samples = call[0][0]
            assert all(
                s.name in staleness_names for s in samples
            ), f"Only staleness signals expected on failure, got: {samples}"


# ===================================================================
# Real async timing tests
# ===================================================================


class TestRealAsyncTiming:
    """Tests that exercise MetricCollectionLoop with real asyncio.sleep timing."""

    @pytest.mark.anyio
    async def test_collector_called_at_expected_interval(self) -> None:
        """With interval=0.05s and running for ~0.3s, collector should be called ~6 times."""
        collector = _PassCollector(metrics=[GaugeSample(name="m", labels={}, value=1.0)])
        collector.collect_interval = 0.05
        exporter = _make_exporter()
        loop = MetricCollectionLoop(node_id="n0", collectors=[collector], exporter=exporter)

        await loop.start()
        await asyncio.sleep(0.35)
        await loop.stop()

        assert 4 <= collector.collect_count <= 10

    @pytest.mark.anyio
    async def test_failing_collector_does_not_block_healthy_collector(self) -> None:
        """One failing collector should not prevent another from collecting."""
        healthy = _PassCollector(metrics=[GaugeSample(name="h", labels={}, value=1.0)])
        healthy.collect_interval = 0.05
        failing = FailingCollector(collect_interval=0.05)
        exporter = _make_exporter()
        loop = MetricCollectionLoop(
            node_id="n0",
            collectors=[failing, healthy],
            exporter=exporter,
        )

        await loop.start()
        await asyncio.sleep(0.3)
        await loop.stop()

        assert healthy.collect_count >= 3
        assert failing.call_count >= 3


class TestInjectNodeId:
    """Unit tests for _inject_node_id — ensures all collector samples
    carry node_id so Prometheus backend doesn't lack the column that
    node-level detectors depend on."""

    def test_adds_node_id_to_gauge_without_it(self) -> None:
        samples = [GaugeSample(name="gpu_temp", labels={"gpu": "0"}, value=65.0)]
        result = _inject_node_id(samples, node_id="node-1")

        assert len(result) == 1
        assert result[0].labels["node_id"] == "node-1"
        assert result[0].labels["gpu"] == "0"

    def test_adds_node_id_to_counter_without_it(self) -> None:
        samples = [CounterSample(name="xid_count", labels={}, delta=1.0)]
        result = _inject_node_id(samples, node_id="node-1")

        assert result[0].labels["node_id"] == "node-1"

    def test_preserves_existing_node_id(self) -> None:
        samples = [GaugeSample(name="m", labels={"node_id": "original"}, value=1.0)]
        result = _inject_node_id(samples, node_id="injected")

        assert result[0].labels["node_id"] == "original"

    def test_handles_empty_list(self) -> None:
        result = _inject_node_id([], node_id="node-1")

        assert result == []

    def test_multiple_samples_all_get_node_id(self) -> None:
        samples = [
            GaugeSample(name="a", labels={"gpu": "0"}, value=1.0),
            GaugeSample(name="b", labels={"device": "ib0"}, value=0.0),
            CounterSample(name="c", labels={}, delta=1.0),
        ]
        result = _inject_node_id(samples, node_id="node-2")

        assert all(s.labels["node_id"] == "node-2" for s in result)
