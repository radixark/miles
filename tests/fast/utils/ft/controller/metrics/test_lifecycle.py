"""Tests for miles.utils.ft.controller.metrics.lifecycle."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from miles.utils.ft.controller.metrics.lifecycle import start_metric_store_task, stop_metric_store_task


class TestStartMetricStoreTask:
    def test_start_creates_task(self) -> None:
        store = AsyncMock()
        store.start = AsyncMock(side_effect=asyncio.CancelledError)

        async def _run() -> None:
            task = await start_metric_store_task(store)
            assert isinstance(task, asyncio.Task)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())

    def test_start_retries_on_exception(self) -> None:
        """store.start() crashes once, then hangs (simulated via CancelledError)."""
        call_count = 0

        async def _failing_start() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("scrape failed")
            await asyncio.Event().wait()

        store = AsyncMock()
        store.start = _failing_start

        async def _run() -> None:
            import miles.utils.ft.controller.metrics.lifecycle as mod

            original_delay = mod._SCRAPE_RESTART_DELAY_SECONDS
            mod._SCRAPE_RESTART_DELAY_SECONDS = 0.01
            try:
                task = await start_metric_store_task(store)
                await asyncio.sleep(0.1)
                assert call_count >= 2
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            finally:
                mod._SCRAPE_RESTART_DELAY_SECONDS = original_delay

        asyncio.run(_run())


class TestStopMetricStoreTask:
    def test_stop_cancels_task(self) -> None:
        store = AsyncMock()
        store.stop = AsyncMock()

        async def _run() -> None:
            async def _noop() -> None:
                await asyncio.Event().wait()

            task = asyncio.create_task(_noop())
            await stop_metric_store_task(store=store, task=task)

            assert task.cancelled()
            store.stop.assert_awaited_once()

        asyncio.run(_run())

    def test_stop_handles_cancelled_error_gracefully(self) -> None:
        store = AsyncMock()
        store.stop = AsyncMock()

        async def _run() -> None:
            async def _already_done() -> None:
                pass

            task = asyncio.create_task(_already_done())
            await task
            await stop_metric_store_task(store=store, task=task)

        asyncio.run(_run())

    def test_stop_still_cancels_task_when_store_stop_raises(self) -> None:
        """P1 item 14: store.stop() raises → task should still be cancelled."""

        async def _run() -> None:
            store = AsyncMock()
            store.stop = AsyncMock(side_effect=RuntimeError("stop failed"))

            async def _noop() -> None:
                await asyncio.Event().wait()

            task = asyncio.create_task(_noop())

            try:
                await stop_metric_store_task(store=store, task=task)
            except RuntimeError:
                pass

            assert task.cancelled() or task.done()

        asyncio.run(_run())


class TestLifecycleWithImmediateReturnStore:
    @pytest.mark.asyncio
    async def test_immediate_return_start_does_not_busy_loop(self) -> None:
        """When store.start() returns immediately (e.g. PrometheusClient),
        the lifecycle must NOT spin in a tight busy-wait loop. Previously
        start() was a no-op pass, causing the while True loop in
        start_metric_store_task to spin without delay."""
        call_count = 0

        async def _immediate_start() -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                await asyncio.Event().wait()

        store = AsyncMock()
        store.start = _immediate_start

        import miles.utils.ft.controller.metrics.lifecycle as mod

        original_delay = mod._SCRAPE_RESTART_DELAY_SECONDS
        mod._SCRAPE_RESTART_DELAY_SECONDS = 0.01
        try:
            task = await start_metric_store_task(store)
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            mod._SCRAPE_RESTART_DELAY_SECONDS = original_delay

        # Previously, with immediate-return start(), the loop would call
        # start() thousands of times in 0.15s. Now it should be bounded
        # by the restart delay.
        assert call_count < 20, (
            f"start() called {call_count} times — suggests busy-wait loop"
        )


class TestScrapeLoopMaxRestarts:
    def test_stops_after_max_restarts_exceeded(self) -> None:
        """Previously the scrape loop retried forever on persistent failures,
        flooding logs without escalation. Now it stops after _MAX_SCRAPE_RESTARTS."""
        call_count = 0

        async def _always_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("persistent failure")

        store = AsyncMock()
        store.start = _always_fail

        async def _run() -> None:
            import miles.utils.ft.controller.metrics.lifecycle as mod

            original_delay = mod._SCRAPE_RESTART_DELAY_SECONDS
            original_max = mod._MAX_SCRAPE_RESTARTS
            mod._SCRAPE_RESTART_DELAY_SECONDS = 0.001
            mod._MAX_SCRAPE_RESTARTS = 3
            try:
                task = await start_metric_store_task(store)
                await asyncio.sleep(0.5)
                assert task.done()
                assert call_count == 3
            finally:
                mod._SCRAPE_RESTART_DELAY_SECONDS = original_delay
                mod._MAX_SCRAPE_RESTARTS = original_max

        asyncio.run(_run())
