"""Tests for the controller state machine (MainState + handlers)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from tests.fast.utils.ft.utils.controller_fakes import FakeMainJob, FakeNodeManager, FakeNotifier
from tests.fast.utils.ft.utils.diagnostic_fakes import FakeDiagnosticOrchestrator

from miles.utils.ft.adapters.types import JobStatus, SubsystemActuatorProtocol
from miles.utils.ft.controller.metrics.exporter import NullControllerExporter
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.main import (
    MainContext,
    NormalSt,
    RestartingMainJobSt,
    create_main_stepper,
)
from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomalySt,
    RecoveringSt,
    SubsystemState,
)
from miles.utils.ft.controller.state_machines.recovery.models import EvictingAndRestartingSt, StopTimeDiagnosticsSt
from miles.utils.ft.controller.state_machines.restart.models import (
    ExternalExecutionResult,
    ExternalRestartingMainJobSt,
)
from miles.utils.ft.controller.subsystem_hub import SubsystemConfig
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.types import MetricStore, TriggerType
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter, SlidingWindowThrottle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frozen_recovering_state() -> RecoveringSt:
    """Create a Recovering state matching what NormalStateHandler would freeze."""
    return RecoveringSt(
        recovery=EvictingAndRestartingSt(
            restart=ExternalRestartingMainJobSt(),
            failed_next_state=StopTimeDiagnosticsSt(),
        ),
        trigger=TriggerType.CRASH,
        recovery_start_time=datetime.now(timezone.utc),
    )


def _make_subsystem_config() -> SubsystemConfig:
    return SubsystemConfig(actuator=AsyncMock(spec=SubsystemActuatorProtocol))


def _make_controller_context(
    *,
    main_job: FakeMainJob | None = None,
    subsystem_configs: dict[str, SubsystemConfig] | None = None,
) -> MainContext:
    resolved_main_job = main_job or FakeMainJob()
    return MainContext(
        main_job=resolved_main_job,
        subsystem_configs=subsystem_configs or {
            "training": _make_subsystem_config(),
        },
        tick_count=10,
        job_status=JobStatus.RUNNING,
        metric_store=MetricStore(
            time_series_store=MiniPrometheus(config=MiniPrometheusConfig()),
            mini_wandb=MiniWandb(),
        ),
        agents={},
        notifier=FakeNotifier(),
        node_manager=FakeNodeManager(),
        diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        cooldown=SlidingWindowThrottle(window_minutes=30.0, max_count=3),
        detector_crash_tracker=SlidingWindowCounter(window_seconds=1800, threshold=5),
        recovery_timeout_seconds=1800,
        max_simultaneous_bad_nodes=3,
        on_new_run=None,
        rank_pids_provider=None,
        controller_exporter=NullControllerExporter(),
        on_recovery_duration=None,
        registration_grace_ticks=5,
    )


async def _step_last(stepper, state, ctx):
    result = None
    async for result in stepper(state, ctx):
        pass
    return result


# ---------------------------------------------------------------------------
# Scenario 1: Normal operation — no state transition
# ---------------------------------------------------------------------------


class TestNormalOperation:
    @pytest.mark.asyncio
    async def test_all_subsystems_detecting_no_transition(self) -> None:
        """Two subsystems in DetectingAnomaly → NormalStateHandler steps them → no transition."""
        stepper = create_main_stepper()
        subsystems: dict[str, SubsystemState] = {
            "training": DetectingAnomalySt(),
            "rollout_0": DetectingAnomalySt(),
        }
        state = NormalSt(subsystems=subsystems)
        context = _make_controller_context(
            subsystem_configs={
                "training": _make_subsystem_config(),
                "rollout_0": _make_subsystem_config(),
            },
        )

        result = await _step_last(stepper, state, context)

        assert result is None


# ---------------------------------------------------------------------------
# Scenario 2: Single subsystem escalation
# ---------------------------------------------------------------------------


class TestSingleSubsystemEscalation:
    @pytest.mark.asyncio
    async def test_nested_restart_request_triggers_escalation(self) -> None:
        """Subsystem in Recovering(EvictingAndRestarting(RestartingMainJob(external_execution_result=None)))
        → peek-and-freeze → stop+submit main job → RestartingMainJobState."""
        main_job = FakeMainJob()
        stepper = create_main_stepper()

        nested_state = _make_frozen_recovering_state()
        subsystems: dict[str, SubsystemState] = {
            "rollout_0": nested_state,
        }
        state = NormalSt(subsystems=subsystems)
        context = _make_controller_context(
            main_job=main_job,
            subsystem_configs={"rollout_0": _make_subsystem_config()},
        )

        result = await _step_last(stepper, state, context)

        assert isinstance(result, RestartingMainJobSt)
        assert result.requestor_name == "rollout_0"
        assert isinstance(result.requestor_frozen_state, RecoveringSt)
        assert main_job._stopped
        assert main_job._submitted


# ---------------------------------------------------------------------------
# Scenario 3: Job restart complete → reset to NormalState
# ---------------------------------------------------------------------------


class TestJobRestartComplete:
    @pytest.mark.asyncio
    async def test_running_status_rebuilds_subsystems_and_signals_requestor(self) -> None:
        """RestartingMainJobState + RUNNING → fresh states → requestor gets restored frozen state."""
        main_job = FakeMainJob(status_sequence=[JobStatus.RUNNING])
        stepper = create_main_stepper()

        frozen = _make_frozen_recovering_state()
        state = RestartingMainJobSt(
            requestor_name="rollout_0",
            start_time=datetime.now(timezone.utc),
            requestor_frozen_state=frozen,
        )
        context = _make_controller_context(
            main_job=main_job,
            subsystem_configs={
                "training": _make_subsystem_config(),
                "rollout_0": _make_subsystem_config(),
            },
        )

        result = await _step_last(stepper, state, context)

        assert isinstance(result, NormalSt)
        assert set(result.subsystems.keys()) == {"training", "rollout_0"}

        # Requestor gets restored frozen state with external_execution_result=SUCCEEDED
        requestor_state = result.subsystems["rollout_0"]
        assert isinstance(requestor_state, RecoveringSt)
        assert isinstance(requestor_state.recovery, EvictingAndRestartingSt)
        assert isinstance(requestor_state.recovery.restart, ExternalRestartingMainJobSt)
        assert requestor_state.recovery.restart.external_execution_result == ExternalExecutionResult.SUCCEEDED

        assert isinstance(result.subsystems["training"], DetectingAnomalySt)


# ---------------------------------------------------------------------------
# Scenario 4: Multiple subsystems, one escalates
# ---------------------------------------------------------------------------


class TestMultiSubsystemOneEscalates:
    @pytest.mark.asyncio
    async def test_only_one_escalating_triggers_single_restart(self) -> None:
        """Two subsystems, only rollout_0 requesting restart → one job restart."""
        main_job = FakeMainJob()
        stepper = create_main_stepper()

        nested_state = _make_frozen_recovering_state()
        subsystems: dict[str, SubsystemState] = {
            "training": DetectingAnomalySt(),
            "rollout_0": nested_state,
        }
        state = NormalSt(subsystems=subsystems)

        context = _make_controller_context(
            main_job=main_job,
            subsystem_configs={
                "training": _make_subsystem_config(),
                "rollout_0": _make_subsystem_config(),
            },
        )

        result = await _step_last(stepper, state, context)

        assert isinstance(result, RestartingMainJobSt)
        assert result.requestor_name == "rollout_0"
        assert main_job._submit_call_count == 1


# ---------------------------------------------------------------------------
# Scenario 5: Job restart pending — keep waiting
# ---------------------------------------------------------------------------


class TestJobRestartPending:
    @pytest.mark.asyncio
    async def test_pending_status_returns_none(self) -> None:
        """RestartingMainJobState + PENDING → handler returns None, stay waiting."""
        main_job = FakeMainJob(status_sequence=[JobStatus.PENDING])
        stepper = create_main_stepper()

        frozen = _make_frozen_recovering_state()
        state = RestartingMainJobSt(
            requestor_name="rollout_0",
            start_time=datetime.now(timezone.utc),
            requestor_frozen_state=frozen,
        )
        context = _make_controller_context(main_job=main_job)

        result = await _step_last(stepper, state, context)

        assert result is None

    @pytest.mark.asyncio
    async def test_failed_status_restores_requestor_with_failed_result(self) -> None:
        """RestartingMainJobState + FAILED → requestor restored with FAILED result."""
        main_job = FakeMainJob(status_sequence=[JobStatus.FAILED])
        stepper = create_main_stepper()

        frozen = _make_frozen_recovering_state()
        state = RestartingMainJobSt(
            requestor_name="rollout_0",
            start_time=datetime.now(timezone.utc),
            requestor_frozen_state=frozen,
        )
        context = _make_controller_context(
            main_job=main_job,
            subsystem_configs={
                "training": _make_subsystem_config(),
                "rollout_0": _make_subsystem_config(),
            },
        )

        result = await _step_last(stepper, state, context)

        assert isinstance(result, NormalSt)
        requestor_state = result.subsystems["rollout_0"]
        assert isinstance(requestor_state, RecoveringSt)
        assert isinstance(requestor_state.recovery, EvictingAndRestartingSt)
        assert isinstance(requestor_state.recovery.restart, ExternalRestartingMainJobSt)
        assert requestor_state.recovery.restart.external_execution_result == ExternalExecutionResult.FAILED

    @pytest.mark.asyncio
    async def test_timeout_writes_timeout_result(self) -> None:
        """RestartingMainJobState + PENDING past timeout → requestor restored with TIMEOUT result."""
        from datetime import timedelta

        main_job = FakeMainJob(status_sequence=[JobStatus.PENDING])
        stepper = create_main_stepper()

        frozen = _make_frozen_recovering_state()
        state = RestartingMainJobSt(
            requestor_name="rollout_0",
            start_time=datetime.now(timezone.utc) - timedelta(seconds=3600),
            requestor_frozen_state=frozen,
        )
        context = _make_controller_context(
            main_job=main_job,
            subsystem_configs={
                "training": _make_subsystem_config(),
                "rollout_0": _make_subsystem_config(),
            },
        )

        result = await _step_last(stepper, state, context)

        assert isinstance(result, NormalSt)
        requestor_state = result.subsystems["rollout_0"]
        assert isinstance(requestor_state, RecoveringSt)
        assert isinstance(requestor_state.recovery, EvictingAndRestartingSt)
        assert isinstance(requestor_state.recovery.restart, ExternalRestartingMainJobSt)
        assert requestor_state.recovery.restart.external_execution_result == ExternalExecutionResult.TIMEOUT
