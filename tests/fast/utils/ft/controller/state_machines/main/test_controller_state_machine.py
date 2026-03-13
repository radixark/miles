"""Tests for the controller state machine (MainState + handlers)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from tests.fast.utils.ft.utils.controller_fakes import (
    FakeMainJob,
    FakeNodeManager,
    FakeNotifier,
    make_failing_main_job,
)
from tests.fast.utils.ft.utils.diagnostic_fakes import FakeDiagnosticOrchestrator

from miles.utils.ft.adapters.types import JobStatus, SubsystemActuatorProtocol
from miles.utils.ft.controller.metrics.exporter import NullControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.main import (
    MainContext,
    NormalSt,
    RestartingMainJobSt,
    create_main_stepper,
)
from miles.utils.ft.controller.state_machines.recovery.models import EvictingAndRestartingSt, StopTimeDiagnosticsSt
from miles.utils.ft.controller.state_machines.restart.models import (
    ExternalExecutionResult,
    ExternalRestartingMainJobSt,
)
from miles.utils.ft.controller.state_machines.subsystem.models import DetectingAnomalySt, RecoveringSt, SubsystemState
from miles.utils.ft.controller.subsystem_hub import SubsystemConfig, SubsystemRuntime, SubsystemSpec
from miles.utils.ft.controller.types import MetricStore, SharedDeps, TriggerType
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter

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


def _make_subsystem_spec() -> SubsystemSpec:
    return SubsystemSpec(
        config=SubsystemConfig(),
        runtime=SubsystemRuntime(actuator=AsyncMock(spec=SubsystemActuatorProtocol)),
    )


def _make_controller_context(
    *,
    main_job: FakeMainJob | None = None,
    subsystem_specs: dict[str, SubsystemSpec] | None = None,
    notifier: FakeNotifier | None = None,
) -> MainContext:
    shared = SharedDeps(
        main_job=main_job or FakeMainJob(),
        subsystem_specs=subsystem_specs
        or {
            "training": _make_subsystem_spec(),
        },
        metric_store=MetricStore(
            time_series_store=MiniPrometheus(config=MiniPrometheusConfig()),
            mini_wandb=MiniWandb(),
        ),
        notifier=notifier or FakeNotifier(),
        node_manager=FakeNodeManager(),
        diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        detector_crash_tracker=SlidingWindowCounter(window_seconds=1800, threshold=5),
        recovery_timeout_seconds=1800,
        max_simultaneous_bad_nodes=3,
        on_main_job_new_run=None,
        rank_pids_provider=None,
        controller_exporter=NullControllerExporter(),
        on_recovery_duration=None,
        registration_grace_ticks=5,
    )
    return MainContext(
        shared=shared,
        tick_count=10,
        run_start_tick=0,
        job_status=JobStatus.RUNNING,
        node_metadata={},
    )


async def _step_last(stepper, state, ctx):
    result = None
    async for next_result in stepper(state, ctx):
        result = next_result
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
            subsystem_specs={
                "training": _make_subsystem_spec(),
                "rollout_0": _make_subsystem_spec(),
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
            subsystem_specs={"rollout_0": _make_subsystem_spec()},
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
            subsystem_specs={
                "training": _make_subsystem_spec(),
                "rollout_0": _make_subsystem_spec(),
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
            subsystem_specs={
                "training": _make_subsystem_spec(),
                "rollout_0": _make_subsystem_spec(),
            },
        )

        result = await _step_last(stepper, state, context)

        assert isinstance(result, RestartingMainJobSt)
        assert result.requestor_name == "rollout_0"
        assert main_job._submit_call_count == 1


# ---------------------------------------------------------------------------
# Scenario 4b: Non-requestor recovery discarded
# ---------------------------------------------------------------------------


class TestNonRequestorRecoveryDiscarded:
    @pytest.mark.asyncio
    async def test_logs_warning_for_non_requestor_in_recovery(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Previously, when one subsystem triggered a main job restart, all
        other subsystems' recovery states were silently discarded. Now a
        warning is logged for each non-requestor subsystem that was in
        recovery so operators can investigate potential cascading issues.
        """
        import logging

        main_job = FakeMainJob()
        stepper = create_main_stepper()

        requestor_state = _make_frozen_recovering_state()
        bystander_state = RecoveringSt(
            recovery=EvictingAndRestartingSt(
                restart=ExternalRestartingMainJobSt(
                    external_execution_result=ExternalExecutionResult.SUCCEEDED,
                ),
                failed_next_state=StopTimeDiagnosticsSt(),
            ),
            trigger=TriggerType.HANG,
            recovery_start_time=datetime.now(timezone.utc),
        )

        subsystems: dict[str, SubsystemState] = {
            "training": bystander_state,
            "rollout_0": requestor_state,
        }
        state = NormalSt(subsystems=subsystems)
        context = _make_controller_context(
            main_job=main_job,
            subsystem_specs={
                "training": _make_subsystem_spec(),
                "rollout_0": _make_subsystem_spec(),
            },
        )

        with caplog.at_level(logging.WARNING, logger="miles.utils.ft.controller.state_machines.main.handlers"):
            await _step_last(stepper, state, context)

        assert "subsystem_recovery_discarded" in caplog.text
        assert "training" in caplog.text

    @pytest.mark.asyncio
    async def test_non_requestor_recovery_state_reset_to_detecting_after_restart(self) -> None:
        """Known limitation: when one subsystem requests main job restart, other
        subsystems' recovery progress is discarded. After the restart completes,
        non-requestor subsystems are reset to DetectingAnomalySt and must
        re-detect faults from scratch."""
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
            subsystem_specs={
                "training": _make_subsystem_spec(),
                "rollout_0": _make_subsystem_spec(),
            },
        )

        result = await _step_last(stepper, state, context)

        assert isinstance(result, NormalSt)
        assert isinstance(result.subsystems["training"], DetectingAnomalySt)
        assert isinstance(result.subsystems["rollout_0"], RecoveringSt)


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
            subsystem_specs={
                "training": _make_subsystem_spec(),
                "rollout_0": _make_subsystem_spec(),
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
            subsystem_specs={
                "training": _make_subsystem_spec(),
                "rollout_0": _make_subsystem_spec(),
            },
        )

        result = await _step_last(stepper, state, context)

        assert isinstance(result, NormalSt)
        requestor_state = result.subsystems["rollout_0"]
        assert isinstance(requestor_state, RecoveringSt)
        assert isinstance(requestor_state.recovery, EvictingAndRestartingSt)
        assert isinstance(requestor_state.recovery.restart, ExternalRestartingMainJobSt)
        assert requestor_state.recovery.restart.external_execution_result == ExternalExecutionResult.TIMEOUT


# ---------------------------------------------------------------------------
# Subsystem stepping order
# ---------------------------------------------------------------------------


class TestSubsystemKeysSyncAssert:
    @pytest.mark.asyncio
    async def test_mismatched_subsystem_keys_raises_assertion_error(self) -> None:
        """NormalHandler.step assumes subsystem_specs keys match
        state.subsystems keys. If they diverge, a KeyError would occur
        deep in the handler. An explicit assert catches this early."""
        stepper = create_main_stepper()
        state = NormalSt(
            subsystems={
                "training": DetectingAnomalySt(),
            }
        )
        context = _make_controller_context(
            subsystem_specs={
                "training": _make_subsystem_spec(),
                "rollout_0": _make_subsystem_spec(),
            },
        )

        with pytest.raises(AssertionError, match="subsystem keys out of sync"):
            await _step_last(stepper, state, context)


class TestSubsystemSteppingOrder:
    @pytest.mark.asyncio
    async def test_subsystems_stepped_in_sorted_order(self) -> None:
        """Previously dict iteration order was used for subsystem stepping,
        making behavior depend on insertion order. Now subsystems are stepped
        in sorted name order for determinism.

        We verify by inspecting which subsystem config objects are passed
        to build_subsystem_context in what order.
        """
        from unittest.mock import patch

        import miles.utils.ft.controller.state_machines.main.subsystem_runner as runner_mod

        call_order: list[str] = []
        specs = {}
        for name in ["zz_last", "aa_first", "mm_middle"]:
            specs[name] = _make_subsystem_spec()
        subsystems = {name: DetectingAnomalySt() for name in specs}

        original_build = runner_mod.build_subsystem_context

        def tracking_build(*, spec, **kwargs):
            name = next(n for n, s in specs.items() if s is spec)
            call_order.append(name)
            return original_build(spec=spec, **kwargs)

        state = NormalSt(subsystems=subsystems)
        context = _make_controller_context(subsystem_specs=specs)

        with patch.object(runner_mod, "build_subsystem_context", tracking_build):
            stepper = create_main_stepper()
            async for _ in stepper(state, context):
                pass

        assert call_order == ["aa_first", "mm_middle", "zz_last"]


# ---------------------------------------------------------------------------
# H-3: _check_main_job_restart uses stop_and_submit — error handling
# ---------------------------------------------------------------------------


class TestMainJobRestartStopFailure:
    @pytest.mark.asyncio
    async def test_stop_failure_sets_failed_result_and_notifies(self) -> None:
        """When stop_and_submit fails, the requestor's state must be advanced
        to ExternalExecutionResult.FAILED so recovery/restart can proceed.
        Previously the handler returned None, leaving the requestor stuck in
        ExternalRestartingMainJobSt(external_execution_result=None) and
        causing repeated stop/start attempts every tick."""
        main_job = make_failing_main_job(
            fail_stop=True,
            status_sequence=[JobStatus.RUNNING],
        )
        notifier = FakeNotifier()
        stepper = create_main_stepper()

        nested_state = _make_frozen_recovering_state()
        subsystems: dict[str, SubsystemState] = {"rollout_0": nested_state}
        state = NormalSt(subsystems=subsystems)
        context = _make_controller_context(
            main_job=main_job,
            subsystem_specs={"rollout_0": _make_subsystem_spec()},
            notifier=notifier,
        )

        result = await _step_last(stepper, state, context)

        # Step 1: Verify we get a NormalSt back (not None)
        assert isinstance(result, NormalSt)

        # Step 2: Requestor state is advanced with FAILED result
        requestor_state = result.subsystems["rollout_0"]
        assert isinstance(requestor_state, RecoveringSt)
        assert isinstance(requestor_state.recovery, EvictingAndRestartingSt)
        assert isinstance(requestor_state.recovery.restart, ExternalRestartingMainJobSt)
        assert requestor_state.recovery.restart.external_execution_result == ExternalExecutionResult.FAILED

        # Step 3: No actual job submission occurred
        assert not main_job._submitted

        # Step 4: Notification was sent
        assert len(notifier.calls) >= 1
        assert any("stop_and_submit failed" in call[1] for call in notifier.calls)

    @pytest.mark.asyncio
    async def test_stop_failure_does_not_trigger_repeated_restart_next_tick(self) -> None:
        """Verify the fix eliminates repeated stop/start: after stop_and_submit
        failure, _find_restart_requestor returns None because the requestor's
        external_execution_result is now FAILED (not None)."""
        main_job = make_failing_main_job(
            fail_stop=True,
            status_sequence=[JobStatus.RUNNING],
        )
        stepper = create_main_stepper()

        nested_state = _make_frozen_recovering_state()
        subsystems: dict[str, SubsystemState] = {"rollout_0": nested_state}
        state = NormalSt(subsystems=subsystems)
        context = _make_controller_context(
            main_job=main_job,
            subsystem_specs={"rollout_0": _make_subsystem_spec()},
        )

        result = await _step_last(stepper, state, context)
        assert isinstance(result, NormalSt)

        # Simulating next tick: _find_restart_requestor should not find a
        # requestor since external_execution_result is now FAILED
        from miles.utils.ft.controller.state_machines.main.handlers import _find_restart_requestor

        assert _find_restart_requestor(result.subsystems) is None
