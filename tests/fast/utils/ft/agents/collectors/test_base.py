"""Tests for miles.utils.ft.agents.collectors.base."""

from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft.agents.collectors.base import BaseCollector, CollectorOutput
from miles.utils.ft.agents.types import GaugeSample, MetricSample


class _SuccessCollector(BaseCollector):
    collect_interval: float = 1.0

    def _collect_sync(self) -> list[MetricSample]:
        return [GaugeSample(name="test_metric", labels={}, value=42.0)]


class _SlowCollector(BaseCollector):
    collect_interval: float = 0.1

    def _collect_sync(self) -> list[MetricSample]:
        import time

        time.sleep(5)
        return []


class _CrashingCollector(BaseCollector):
    collect_interval: float = 1.0

    def _collect_sync(self) -> list[MetricSample]:
        raise RuntimeError("collector crashed")


class TestCollectIntervalValidation:
    def test_negative_collect_interval_on_subclass_raises(self) -> None:
        """Previously negative collect_interval made the internal timeout
        (interval * 2) negative, causing asyncio.wait_for to raise
        ValueError. Now rejected at class definition time."""
        with pytest.raises(ValueError, match="collect_interval must be >= 0"):

            class _BadCollector(BaseCollector):
                collect_interval: float = -1.0

                def _collect_sync(self) -> list[MetricSample]:
                    return []

    def test_zero_collect_interval_allowed(self) -> None:
        class _ZeroCollector(BaseCollector):
            collect_interval: float = 0.0

            def _collect_sync(self) -> list[MetricSample]:
                return []

        assert _ZeroCollector.collect_interval == 0.0


class TestBaseCollector:
    def test_collect_returns_collector_output(self) -> None:
        collector = _SuccessCollector()
        result = asyncio.run(collector.collect())

        assert isinstance(result, CollectorOutput)
        assert len(result.metrics) == 1
        assert result.metrics[0].name == "test_metric"

    def test_collect_timeout_raises_timeout_error(self) -> None:
        """Timeout = collect_interval * 2, so 0.1 * 2 = 0.2s should fire before the 5s sleep."""
        collector = _SlowCollector()

        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(collector.collect())

    def test_collect_sync_exception_propagates(self) -> None:
        collector = _CrashingCollector()

        with pytest.raises(RuntimeError, match="collector crashed"):
            asyncio.run(collector.collect())

    def test_close_is_noop_by_default(self) -> None:
        collector = _SuccessCollector()

        asyncio.run(collector.close())
