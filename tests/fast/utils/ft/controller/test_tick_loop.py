"""Unit tests for TickLoop (P0 item 1)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.metrics.exporter import NullControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.node_agents.registry import NodeAgentRegistry
from miles.utils.ft.controller.state_machines.main.models import MainState, NormalSt
from miles.utils.ft.controller.state_machines.subsystem import RecoveringSt
from miles.utils.ft.controller.state_machines.recovery.models import RECOVERY_STATE_TO_INT, RealtimeChecksSt
from miles.utils.ft.controller.tick_loop import TickLoop
from miles.utils.ft.controller.types import MetricStore, TriggerType
from miles.utils.ft.utils.box import Box
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter, SlidingWindowThrottle

from tests.fast.utils.ft.utils.controller_fakes import FakeMainJob, FakeNotifier

pytestmark = pytest.mark.anyio


def _make_tick_loop(
    *,
    main_job: FakeMainJob | None = None,
    notifier: FakeNotifier | None = None,
    state_machine: MagicMock | None = None,
    registration_grace_ticks: int = 0,
) -> TickLoop:
    if main_job is None:
        main_job = FakeMainJob(status_sequence=[JobStatus.RUNNING])
    if state_machine is None:
        state_machine = MagicMock()
        state_machine.step = AsyncMock()
        state_machine.state = NormalSt(subsystems={})

    return TickLoop(
        state_machine=state_machine,
        training_rank_roster_box=Box(None),
        node_agent_registry=NodeAgentRegistry(),
        main_job=main_job,
        metric_store=MetricStore(
            time_series_store=MiniPrometheus(config=MiniPrometheusConfig()),
            mini_wandb=MiniWandb(),
        ),
        notifier=notifier,
        node_manager=MagicMock(),
        cooldown=SlidingWindowThrottle(window_minutes=30, max_count=3),
        max_simultaneous_bad_nodes=2,
        diagnostic_orchestrator=MagicMock(),
        recovery_timeout_seconds=600,
        subsystem_configs={},
        registration_grace_ticks=registration_grace_ticks,
    )


class TestTickExceptionTriggersNotification:
    async def test_tick_exception_records_failure(self) -> None:
        """tick() exception → _tick_failure_tracker.record() is called."""
        sm = MagicMock()
        sm.step = AsyncMock(side_effect=RuntimeError("boom"))
        sm.state = NormalSt(subsystems={})

        loop = _make_tick_loop(state_machine=sm)
        await loop.tick()

        assert loop._tick_failure_tracker._counter > 0

    async def test_tick_persistent_failure_triggers_notification(self) -> None:
        """Exceeding threshold triggers safe_notify."""
        sm = MagicMock()
        sm.step = AsyncMock(side_effect=RuntimeError("boom"))
        sm.state = NormalSt(subsystems={})
        notifier = FakeNotifier()

        loop = _make_tick_loop(state_machine=sm, notifier=notifier)
        loop._tick_failure_tracker = SlidingWindowCounter(window_seconds=300, threshold=2)

        for _ in range(3):
            await loop.tick()

        assert len(notifier.calls) > 0
        assert "persistently failing" in notifier.calls[0][0].lower()


class TestUpdateExporterMetrics:
    async def test_recovery_state_maps_to_phase_int(self) -> None:
        """RecoveringSt → phase int from RECOVERY_STATE_TO_INT mapping."""
        from datetime import datetime, timezone

        recovery_state = RealtimeChecksSt()
        subsystem_state = RecoveringSt(
            recovery=recovery_state,
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
        )

        sm = MagicMock()
        sm.step = AsyncMock()
        sm.state = NormalSt(subsystems={"training": subsystem_state})

        loop = _make_tick_loop(state_machine=sm)
        exporter = MagicMock()
        loop._controller_exporter = exporter

        await loop.tick()

        exporter.update_from_state.assert_called_once()
        call_kwargs = exporter.update_from_state.call_args.kwargs
        assert call_kwargs["recovery_phase_int"] == RECOVERY_STATE_TO_INT[RealtimeChecksSt]

    async def test_non_recovery_state_maps_to_phase_zero(self) -> None:
        """Non-RecoveringSt → phase int 0."""
        sm = MagicMock()
        sm.step = AsyncMock()
        sm.state = NormalSt(subsystems={})

        loop = _make_tick_loop(state_machine=sm)
        exporter = MagicMock()
        loop._controller_exporter = exporter

        await loop.tick()

        exporter.update_from_state.assert_called_once()
        call_kwargs = exporter.update_from_state.call_args.kwargs
        assert call_kwargs["recovery_phase_int"] == 0


class TestExtractMainState:
    def test_returns_training_subsystem_for_normal_state(self) -> None:
        sm = MagicMock()
        training_state = MagicMock()
        sm.state = NormalSt(subsystems={"training": training_state})

        loop = _make_tick_loop(state_machine=sm)
        result = loop._extract_main_state()
        assert result is training_state

    def test_returns_none_for_non_normal_state(self) -> None:
        sm = MagicMock()
        sm.state = MagicMock(spec=MainState)

        loop = _make_tick_loop(state_machine=sm)
        result = loop._extract_main_state()
        assert result is None


class TestRegistrationGraceTicks:
    async def test_roster_warn_and_coverage_skipped_when_roster_is_none(self) -> None:
        """When roster is None, warn_if_incomplete and coverage check are skipped."""
        sm = MagicMock()
        sm.step = AsyncMock()
        sm.state = NormalSt(subsystems={})

        loop = _make_tick_loop(state_machine=sm, registration_grace_ticks=5)
        coverage_checker = MagicMock()
        loop._node_agent_coverage_checker = coverage_checker

        await loop.tick()

        coverage_checker.check.assert_not_called()
