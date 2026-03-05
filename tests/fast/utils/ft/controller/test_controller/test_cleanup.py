"""Tests for FtController.run() cleanup paths and _execute_decision defensive branches."""
from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft.models import ActionType, Decision
from tests.fast.utils.ft.conftest import (
    FakeNotifier,
    FixedDecisionDetector,
    make_test_controller,
)


class TestRunCleanupNotifierAclose:
    """Verify controller.run() handles notifier.aclose() exceptions gracefully."""

    @pytest.mark.anyio
    async def test_notifier_aclose_exception_does_not_propagate(self) -> None:
        harness = make_test_controller(tick_interval=0.01)

        assert harness.notifier is not None

        async def failing_aclose() -> None:
            raise RuntimeError("webhook connection broken")

        harness.notifier.aclose = failing_aclose  # type: ignore[assignment]

        async def _shutdown_soon() -> None:
            await asyncio.sleep(0.03)
            await harness.controller.shutdown()

        task = asyncio.create_task(_shutdown_soon())
        await harness.controller.run()
        await task

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

        async def _shutdown_soon() -> None:
            await asyncio.sleep(0.03)
            await harness.controller.shutdown()

        task = asyncio.create_task(_shutdown_soon())
        await harness.controller.run()
        await task

        assert stop_called


class TestRunLoopSurviveTickFailure:
    """Verify that run() continues ticking after _tick_inner raises."""

    @pytest.mark.anyio
    async def test_run_continues_after_tick_inner_exception(self) -> None:
        harness = make_test_controller(tick_interval=0.01)

        original_tick_inner = harness.controller._tick_inner
        call_count = 0

        async def _exploding_tick_inner() -> None:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient failure")
            await original_tick_inner()

        harness.controller._tick_inner = _exploding_tick_inner  # type: ignore[assignment]

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

        async def _shutdown_soon() -> None:
            await asyncio.sleep(0.03)
            await harness.controller.shutdown()

        task = asyncio.create_task(_shutdown_soon())
        await harness.controller.run()
        await task

        assert stop_called


class TestExecuteDecisionUnknownAction:
    """Verify _execute_decision raises ValueError for unknown action types."""

    @pytest.mark.anyio
    async def test_unknown_action_type_raises(self) -> None:
        harness = make_test_controller()

        bogus_decision = Decision(action=ActionType.NONE, reason="should not happen")
        object.__setattr__(bogus_decision, "action", "totally_unknown_action")

        with pytest.raises(ValueError, match="Unknown action type"):
            await harness.controller._execute_decision(bogus_decision)
