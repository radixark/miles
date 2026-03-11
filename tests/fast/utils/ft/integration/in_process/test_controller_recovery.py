"""Integration tests: Controller + Recovery state machine full flow.

Each test drives the Controller through multiple ticks to verify
complete decision → recovery → monitoring lifecycle.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from tests.fast.utils.ft.conftest import (
    AlwaysEnterRecoveryDetector,
    FixedDecisionDetector,
    get_sample_value,
    make_test_controller,
    make_test_exporter,
)

import miles.utils.ft.controller.metrics.metric_names as mn
from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.state_machines.main.models import NormalState, RestartingMainJobState
from miles.utils.ft.controller.state_machines.subsystem import Recovering
from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestarting
from miles.utils.ft.controller.state_machines.restart import MonitoringProgress
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType


def _is_monitoring_progress(state: object) -> bool:
    if not isinstance(state, Recovering):
        return False
    recovery = state.recovery
    if isinstance(recovery, EvictingAndRestarting):
        return isinstance(recovery.restart, MonitoringProgress)
    return False


def _disable_detector(detector: FixedDecisionDetector) -> None:
    detector._decision = Decision(action=ActionType.NONE, reason="disabled")


def _inject_iteration_data(harness: object, count: int = 10) -> None:
    active_run_id = harness.controller._training_rank_roster.run_id
    for i in range(1, count + 1):
        harness.mini_wandb.log_step(
            run_id=active_run_id,
            step=i,
            metrics={"iteration": float(i)},
        )


# -------------------------------------------------------------------
# Scenario 1: GPU lost → direct eviction (ENTER_RECOVERY)
# -------------------------------------------------------------------


class TestGpuLostDirectEviction:
    @pytest.mark.anyio
    async def test_gpu_lost_marks_bad_and_restarts(self) -> None:
        detector = FixedDecisionDetector(
            decision=Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-0"],
                reason="GPU unavailable",
                trigger=TriggerType.HARDWARE,
            )
        )
        harness = make_test_controller(
            detectors=[detector],
            status_sequence=[JobStatus.RUNNING],
        )

        await harness.controller._tick()

        assert harness.node_manager.is_node_bad("node-0")
        assert harness.main_job._stopped
        assert harness.main_job._submitted
        assert harness.mini_wandb.latest(metric_name="loss") is None


# -------------------------------------------------------------------
# Scenario 2: Training crash → reattempt → success
# -------------------------------------------------------------------


class TestCrashReattemptSuccess:
    @pytest.mark.anyio
    async def test_crash_reattempt_success(self) -> None:
        enter_recovery = AlwaysEnterRecoveryDetector(reason="training process exited")
        harness = make_test_controller(
            detectors=[enter_recovery],
            status_sequence=[JobStatus.RUNNING],
        )

        # Step 1: first tick chains through recovery to MonitoringProgress
        await harness.controller._tick()
        assert isinstance(harness.controller._training_state_machine.state, Recovering)
        assert _is_monitoring_progress(harness.controller._training_state_machine.state)

        # Step 2: inject iteration progress, disable detector to prevent re-entry
        _disable_detector(enter_recovery)
        _inject_iteration_data(harness)

        await harness.controller._tick()
        assert not isinstance(harness.controller._training_state_machine.state, Recovering)


# -------------------------------------------------------------------
# Scenario 3: Training crash → reattempt fails → diagnose → notify
# -------------------------------------------------------------------


class TestCrashReattemptFailDiagnoseNotify:
    @pytest.mark.anyio
    async def test_crash_diagnose_notify(self) -> None:
        enter_recovery = AlwaysEnterRecoveryDetector(reason="training process exited")
        harness = make_test_controller(
            detectors=[enter_recovery],
            status_sequence=[JobStatus.FAILED],
        )

        # Single tick chains: ENTER_RECOVERY → EvictingAndRestarting → FAILED
        # → StopTimeDiagnostics → all-pass → NotifyHumans → RecoveryDone
        # → detector fires again (cooldown eventually throttles)
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) >= 1


# -------------------------------------------------------------------
# Scenario 4: Global timeout
# -------------------------------------------------------------------


class TestGlobalTimeout:
    @pytest.mark.anyio
    async def test_global_timeout_triggers_notify(self) -> None:
        """When the main job restart stays PENDING past the timeout, a notification is sent."""
        enter_recovery = AlwaysEnterRecoveryDetector(
            trigger=TriggerType.HANG,
            reason="hang detected",
        )
        harness = make_test_controller(
            detectors=[enter_recovery],
            status_sequence=[JobStatus.PENDING] * 100,
        )

        # Step 1: enter recovery → escalates to RestartingMainJobState
        await harness.controller._tick()
        main_state = harness.controller._state_machine.state
        assert isinstance(main_state, RestartingMainJobState)

        # Step 2: inject old start_time to trigger timeout
        _disable_detector(enter_recovery)
        harness.controller._state_machine.force_state(
            RestartingMainJobState(
                requestor_name=main_state.requestor_name,
                start_time=datetime.now(timezone.utc) - timedelta(seconds=1801),
            )
        )

        await harness.controller._tick()
        # After timeout, should return to NormalState
        assert isinstance(harness.controller._state_machine.state, NormalState)
        assert harness.notifier is not None
        assert any("Recovery Alert" in call[0] for call in harness.notifier.calls)


# -------------------------------------------------------------------
# Scenario 5: Recovery complete → back to monitoring mode
# -------------------------------------------------------------------


class TestRecoveryCompleteBackToMonitoring:
    @pytest.mark.anyio
    async def test_back_to_monitoring_after_recovery(self) -> None:
        enter_recovery = AlwaysEnterRecoveryDetector(reason="training process exited")
        harness = make_test_controller(
            detectors=[enter_recovery],
            status_sequence=[JobStatus.RUNNING],
        )

        await harness.controller._tick()
        assert isinstance(harness.controller._training_state_machine.state, Recovering)
        initial_detector_count = enter_recovery.call_count

        # Step 2: complete recovery with iteration data, detector disabled
        _disable_detector(enter_recovery)
        _inject_iteration_data(harness)
        await harness.controller._tick()
        assert not isinstance(harness.controller._training_state_machine.state, Recovering)

        # Step 3: re-enable detector and verify it runs after recovery
        enter_recovery._decision = Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger=TriggerType.CRASH,
            reason="test recovery again",
        )
        harness.controller.training_rank_roster.rank_placement[0] = "node-0"
        await harness.controller._tick()
        assert enter_recovery.call_count > initial_detector_count


# -------------------------------------------------------------------
# Scenario 6: Exporter mode gauge
# -------------------------------------------------------------------


class TestExporterModeGauge:
    @pytest.mark.anyio
    async def test_mode_gauge_during_recovery(self) -> None:
        registry, exporter = make_test_exporter()
        enter_recovery = AlwaysEnterRecoveryDetector(reason="test")
        harness = make_test_controller(
            detectors=[enter_recovery],
            status_sequence=[JobStatus.RUNNING],
            controller_exporter=exporter,
        )

        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 0.0

        await harness.controller._tick()
        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 1.0
        assert get_sample_value(registry, mn.CONTROLLER_RECOVERY_PHASE) is not None
        assert get_sample_value(registry, mn.CONTROLLER_RECOVERY_PHASE) > 0

        # Complete recovery
        _disable_detector(enter_recovery)
        _inject_iteration_data(harness)
        await harness.controller._tick()

        assert not isinstance(harness.controller._training_state_machine.state, Recovering)
        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 0.0
        assert get_sample_value(registry, mn.CONTROLLER_RECOVERY_PHASE) == 0.0
