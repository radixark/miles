"""Unit tests for TickLoop.

TickLoop used to hold all shared dependencies (main_job, metric_store,
notifier, node_manager, etc.) duplicating what FtController already held.
Now TickLoop is a pure execution engine with only tick-specific state,
and shared deps come via TickDeps on each tick() call.
"""
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
from miles.utils.ft.controller.tick_loop import TickDeps, TickLoop
from miles.utils.ft.controller.types import MetricStore, TriggerType
from miles.utils.ft.utils.box import Box
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter

from tests.fast.utils.ft.utils.controller_fakes import FakeMainJob, FakeNotifier

pytestmark = pytest.mark.anyio


def _make_tick_loop(
    *,
    state_machine: MagicMock | None = None,
    registration_grace_ticks: int = 0,
) -> TickLoop:
    if state_machine is None:
        state_machine = MagicMock()
        state_machine.step = AsyncMock()
        state_machine.state = NormalSt(subsystems={})

    return TickLoop(
        state_machine=state_machine,
        registration_grace_ticks=registration_grace_ticks,
    )


def _make_tick_deps(
    *,
    main_job: FakeMainJob | None = None,
    notifier: FakeNotifier | None = None,
    subsystem_specs: dict | None = None,
) -> TickDeps:
    if main_job is None:
        main_job = FakeMainJob(status_sequence=[JobStatus.RUNNING])

    return TickDeps(
        main_job=main_job,
        metric_store=MetricStore(
            time_series_store=MiniPrometheus(config=MiniPrometheusConfig()),
            mini_wandb=MiniWandb(),
        ),
        notifier=notifier,
        node_manager=MagicMock(),
        diagnostic_orchestrator=MagicMock(),
        recovery_timeout_seconds=600,
        max_simultaneous_bad_nodes=2,
        subsystem_specs=subsystem_specs or {},
        on_main_job_new_run=lambda run_id: None,
        rank_pids_provider=lambda node_id: {},
        on_recovery_duration=None,
        controller_exporter=NullControllerExporter(),
        registration_grace_ticks=0,
        training_rank_roster_box=Box(None),
        node_agent_registry=NodeAgentRegistry(),
    )


class TestTickExceptionTriggersNotification:
    async def test_tick_exception_records_failure(self) -> None:
        """tick() exception → _tick_failure_tracker.record() is called."""
        sm = MagicMock()
        sm.step = AsyncMock(side_effect=RuntimeError("boom"))
        sm.state = NormalSt(subsystems={})

        loop = _make_tick_loop(state_machine=sm)
        await loop.tick(_make_tick_deps())

        assert loop._tick_failure_tracker.count > 0

    async def test_tick_persistent_failure_triggers_notification(self) -> None:
        """Exceeding threshold triggers safe_notify."""
        sm = MagicMock()
        sm.step = AsyncMock(side_effect=RuntimeError("boom"))
        sm.state = NormalSt(subsystems={})
        notifier = FakeNotifier()

        loop = _make_tick_loop(state_machine=sm)
        loop._tick_failure_tracker = SlidingWindowCounter(window_seconds=300, threshold=2)
        deps = _make_tick_deps(notifier=notifier)

        for _ in range(3):
            await loop.tick(deps)

        assert len(notifier.calls) > 0
        assert "persistently failing" in notifier.calls[0][0].lower()


class TestUpdateExporterMetrics:
    async def test_recovery_state_maps_to_phase_int_per_subsystem(self) -> None:
        """RecoveringSt subsystem → per-subsystem phase int in subsystem_modes."""
        from datetime import datetime, timezone

        from miles.utils.ft.controller.state_machines.subsystem import DetectingAnomalySt

        recovery_state = RealtimeChecksSt()
        subsystem_state = RecoveringSt(
            recovery=recovery_state,
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
        )

        sm = MagicMock()
        sm.step = AsyncMock()
        sm.state = NormalSt(subsystems={
            "training": subsystem_state,
            "networking": DetectingAnomalySt(),
        })

        loop = _make_tick_loop(state_machine=sm)
        deps = _make_tick_deps()
        exporter = MagicMock()
        deps.controller_exporter = exporter

        await loop.tick(deps)

        exporter.update_from_state.assert_called_once()
        call_kwargs = exporter.update_from_state.call_args.kwargs
        assert call_kwargs["subsystem_modes"]["training"] == (True, RECOVERY_STATE_TO_INT[RealtimeChecksSt])
        assert call_kwargs["subsystem_modes"]["networking"] == (False, 0)

    async def test_non_recovery_state_maps_to_phase_zero(self) -> None:
        """Non-RecoveringSt → phase int 0."""
        sm = MagicMock()
        sm.step = AsyncMock()
        sm.state = NormalSt(subsystems={})

        loop = _make_tick_loop(state_machine=sm)
        deps = _make_tick_deps()
        exporter = MagicMock()
        deps.controller_exporter = exporter

        await loop.tick(deps)

        exporter.update_from_state.assert_called_once()
        call_kwargs = exporter.update_from_state.call_args.kwargs
        assert call_kwargs["subsystem_modes"] == {}


class TestCollectSubsystemModes:
    def test_normal_state_with_subsystems(self) -> None:
        """_collect_subsystem_modes iterates all subsystems."""
        from datetime import datetime, timezone

        from miles.utils.ft.controller.state_machines.subsystem import DetectingAnomalySt

        sm = MagicMock()
        sm.state = NormalSt(subsystems={
            "training": RecoveringSt(
                recovery=RealtimeChecksSt(),
                trigger=TriggerType.CRASH,
                recovery_start_time=datetime.now(timezone.utc),
            ),
            "networking": DetectingAnomalySt(),
        })

        loop = _make_tick_loop(state_machine=sm)
        deps = _make_tick_deps()
        result = loop._collect_subsystem_modes(deps)

        assert result["training"] == (True, RECOVERY_STATE_TO_INT[RealtimeChecksSt])
        assert result["networking"] == (False, 0)

    def test_non_normal_state_returns_all_subsystems_as_idle(self) -> None:
        """When state machine is not in NormalSt, _collect_subsystem_modes
        returns all configured subsystems with (False, 0)."""
        sm = MagicMock()
        sm.state = MagicMock(spec=MainState)

        loop = _make_tick_loop(state_machine=sm)
        deps = _make_tick_deps(subsystem_specs={
            "training": MagicMock(),
            "networking": MagicMock(),
        })

        result = loop._collect_subsystem_modes(deps)

        assert result == {
            "training": (False, 0),
            "networking": (False, 0),
        }


class TestRecoveryStateToIntCompleteness:
    """RECOVERY_STATE_TO_INT uses dict[key] which raises KeyError on unmapped state types."""

    def test_all_recovery_states_are_mapped(self) -> None:
        from miles.utils.ft.controller.state_machines.recovery.models import (
            EvictingAndRestartingSt,
            NotifyHumansSt,
            RecoveryDoneSt,
            StopTimeDiagnosticsSt,
        )

        expected_types = [RealtimeChecksSt, EvictingAndRestartingSt, StopTimeDiagnosticsSt, NotifyHumansSt, RecoveryDoneSt]
        for state_type in expected_types:
            assert state_type in RECOVERY_STATE_TO_INT, f"{state_type.__name__} missing from mapping"


class TestCollectSubsystemModesRestartingMainJob:
    def test_requestor_reported_as_recovery_during_main_job_restart(self) -> None:
        """During RestartingMainJobSt, the requestor's frozen recovery state
        should still appear as recovery in Prometheus metrics."""
        from datetime import datetime, timezone

        from miles.utils.ft.controller.state_machines.main.models import RestartingMainJobSt
        from miles.utils.ft.controller.state_machines.subsystem import RecoveringSt

        frozen = RecoveringSt(
            recovery=RealtimeChecksSt(),
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
        )
        sm = MagicMock()
        sm.state = RestartingMainJobSt(
            requestor_name="rollout_0",
            start_time=datetime.now(timezone.utc),
            requestor_frozen_state=frozen,
        )

        loop = _make_tick_loop(state_machine=sm)
        deps = _make_tick_deps(subsystem_specs={
            "training": MagicMock(),
            "rollout_0": MagicMock(),
        })

        result = loop._collect_subsystem_modes(deps)

        assert result["rollout_0"] == (True, RECOVERY_STATE_TO_INT[RealtimeChecksSt])
        assert result["training"] == (False, 0)


class TestRegistrationGraceTicks:
    async def test_roster_warn_and_coverage_skipped_when_roster_is_none(self) -> None:
        """When roster is None, warn_if_incomplete and coverage check are skipped."""
        sm = MagicMock()
        sm.step = AsyncMock()
        sm.state = NormalSt(subsystems={})

        loop = _make_tick_loop(state_machine=sm, registration_grace_ticks=5)
        deps = _make_tick_deps()
        coverage_checker = MagicMock()
        loop._node_agent_coverage_checker = coverage_checker

        await loop.tick(deps)

        coverage_checker.check.assert_not_called()
