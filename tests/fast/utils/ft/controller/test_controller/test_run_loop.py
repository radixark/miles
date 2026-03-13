"""Tests for controller run loop metric store health checks.

Previously the controller continued ticking even after the scrape loop
died (exhausted restarts), running detectors on stale/empty data.
Now the controller checks MetricStoreTaskHandle health before each tick
and raises RuntimeError if the metric store is unhealthy.
"""

from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft.controller.metrics.lifecycle import MetricStoreTaskHandle
from miles.utils.ft.controller.controller import FtController


class TestCheckMetricStoreHealth:
    def test_healthy_handle_does_not_raise(self) -> None:
        async def _run() -> None:
            task = asyncio.create_task(asyncio.Event().wait())
            handle = MetricStoreTaskHandle(task=task)
            try:
                FtController._check_metric_store_health(handle)
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        asyncio.run(_run())

    def test_raises_when_task_is_done(self) -> None:
        async def _run() -> None:
            async def _finish() -> None:
                pass

            task = asyncio.create_task(_finish())
            await task
            handle = MetricStoreTaskHandle(task=task)

            with pytest.raises(RuntimeError, match="Metric store unhealthy"):
                FtController._check_metric_store_health(handle)

        asyncio.run(_run())

    def test_raises_when_restart_exhausted(self) -> None:
        async def _run() -> None:
            task = asyncio.create_task(asyncio.Event().wait())
            handle = MetricStoreTaskHandle(task=task, restart_exhausted=True)
            try:
                with pytest.raises(RuntimeError, match="Metric store unhealthy"):
                    FtController._check_metric_store_health(handle)
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        asyncio.run(_run())

    def test_error_message_includes_exception_info(self) -> None:
        async def _run() -> None:
            async def _finish() -> None:
                pass

            task = asyncio.create_task(_finish())
            await task
            handle = MetricStoreTaskHandle(
                task=task,
                restart_exhausted=True,
                last_exception=ValueError("scrape crashed"),
                last_failure_at=12345.0,
            )

            with pytest.raises(RuntimeError, match="scrape crashed"):
                FtController._check_metric_store_health(handle)

        asyncio.run(_run())
