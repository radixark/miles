"""Tests for FtController.run() cleanup paths."""

from __future__ import annotations

import asyncio

import pytest

from tests.fast.utils.ft.conftest import make_test_controller, run_controller_briefly


class TestRunCleanupNotifierAclose:
    """Verify controller.run() handles notifier.aclose() exceptions gracefully."""

    @pytest.mark.anyio
    async def test_notifier_aclose_exception_does_not_propagate(self) -> None:
        harness = make_test_controller(tick_interval=0.01)

        assert harness.notifier is not None

        async def failing_aclose() -> None:
            raise RuntimeError("webhook connection broken")

        harness.notifier.aclose = failing_aclose  # type: ignore[assignment]

        await run_controller_briefly(harness)

        assert harness.controller._shutting_down

    @pytest.mark.anyio
    async def test_exporter_stop_called_on_shutdown(self) -> None:
        harness = make_test_controller(tick_interval=0.01)

        stop_called = False
        original_stop = harness.controller_exporter.stop

        def tracking_stop() -> None:
            nonlocal stop_called
            stop_called = True
            original_stop()

        harness.controller_exporter.stop = tracking_stop  # type: ignore[assignment]

        await run_controller_briefly(harness)

        assert stop_called


class TestRunLoopSurviveTickFailure:
    """Verify that run() continues ticking after _tick_inner raises."""

    @pytest.mark.anyio
    async def test_run_continues_after_tick_inner_exception(self) -> None:
        harness = make_test_controller(tick_interval=0.01)

        original_get_status = harness.main_job.get_status
        call_count = 0

        async def _exploding_get_status() -> object:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient failure")
            return await original_get_status()

        harness.main_job.get_status = _exploding_get_status  # type: ignore[assignment]

        async def _shutdown_after_ticks() -> None:
            while harness.controller._tick_count < 4:
                await asyncio.sleep(0.01)
            await harness.controller.shutdown()

        task = asyncio.create_task(_shutdown_after_ticks())
        await harness.controller.run()
        await task

        assert harness.controller._tick_count >= 4
        assert call_count >= 4

    @pytest.mark.anyio
    async def test_metric_store_task_stopped_on_shutdown(self) -> None:
        harness = make_test_controller(tick_interval=0.01)

        stop_called = False
        original_stop = harness.metric_store.stop

        async def tracking_stop() -> None:
            nonlocal stop_called
            stop_called = True
            await original_stop()

        harness.metric_store.stop = tracking_stop  # type: ignore[assignment]

        await run_controller_briefly(harness)

        assert stop_called
