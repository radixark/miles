"""Tests for MetricStoreTaskHandle health checking.

The controller checks handle.is_unhealthy before each tick and raises
RuntimeError if the metric store is unhealthy.
"""

from __future__ import annotations

import asyncio


from miles.utils.ft.controller.metrics.lifecycle import MetricStoreTaskHandle


class TestMetricStoreTaskHandleHealthCheck:
    def test_running_handle_is_healthy(self) -> None:
        async def _run() -> None:
            handle = MetricStoreTaskHandle()
            handle.task = asyncio.create_task(asyncio.Event().wait())
            try:
                assert handle.is_unhealthy is False
            finally:
                handle.task.cancel()
                try:
                    await handle.task
                except asyncio.CancelledError:
                    pass

        asyncio.run(_run())

    def test_done_task_is_unhealthy(self) -> None:
        async def _run() -> None:
            async def _finish() -> None:
                pass

            handle = MetricStoreTaskHandle()
            handle.task = asyncio.create_task(_finish())
            await handle.task

            assert handle.is_unhealthy is True
            assert "Metric store unhealthy" in handle.format_health_error()

        asyncio.run(_run())

    def test_error_message_includes_exception_info(self) -> None:
        async def _run() -> None:
            async def _finish() -> None:
                pass

            handle = MetricStoreTaskHandle()
            handle.task = asyncio.create_task(_finish())
            await handle.task
            handle.last_exception = ValueError("scrape crashed")

            error = handle.format_health_error()
            assert "scrape crashed" in error
            assert "ValueError" in error

        asyncio.run(_run())
