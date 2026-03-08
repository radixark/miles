"""Tests for MetricCollectionLoop — start/stop lifecycle and collector error isolation."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from tests.fast.utils.ft.utils import FailingCloseCollector, FailingCollector

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.utils.metric_collection_loop import MetricCollectionLoop
from miles.utils.ft.models.metrics import GaugeSample


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
    async def test_collect_exception_does_not_kill_loop(self) -> None:
        """A collector that raises on collect() should not stop the loop."""
        collector = FailingCollector(collect_interval=0.01)
        exporter = _make_exporter()
        loop = MetricCollectionLoop(node_id="n0", collectors=[collector], exporter=exporter)

        await loop.start()
        await asyncio.sleep(0.05)
        await loop.stop()

        assert collector.call_count >= 2
        exporter.update_metrics.assert_not_called()
