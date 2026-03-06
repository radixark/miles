"""Integration tests: Controller + RecoveryOrchestrator full flow.

Each test drives the Controller through multiple ticks to verify
complete decision → recovery → monitoring lifecycle.
"""
from __future__ import annotations

import pytest

import miles.utils.ft.models.metric_names as mn
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.models.recovery import RecoveryPhase
from miles.utils.ft.protocols.platform import JobStatus
from tests.fast.utils.ft.conftest import (
    AlwaysEnterRecoveryDetector,
    AlwaysNoneDetector,
    FixedDecisionDetector,
    advance_until_recovery_complete,
    get_sample_value,
    inject_gpu_unavailable,
    make_test_controller,
    make_test_exporter,
)


# -------------------------------------------------------------------
# Scenario 1: GPU lost → direct eviction (MARK_BAD_AND_RESTART)
# -------------------------------------------------------------------


class TestGpuLostDirectEviction:
    @pytest.mark.anyio
    async def test_gpu_lost_marks_bad_and_restarts(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=["node-0"],
            reason="GPU unavailable",
        ))
        harness = make_test_controller(
            detectors=[detector],
            status_sequence=[JobStatus.RUNNING],
        )

        await harness.controller._tick()

        assert harness.node_manager.is_node_bad("node-0")
        assert harness.training_job._stopped
        assert harness.training_job._submitted
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

        await harness.controller._tick()
        assert harness.controller._recovery_manager.in_progress
        orch = harness.controller._recovery_manager._orchestrator

        for _ in range(20):
            if orch.phase == RecoveryPhase.MONITORING:
                break
            await harness.controller._tick()
        assert orch.phase == RecoveryPhase.MONITORING

        active_run_id = harness.controller._rank_roster.run_id
        for i in range(1, 11):
            harness.mini_wandb.log_step(
                run_id=active_run_id, step=i,
                metrics={"iteration": float(i)},
            )

        await harness.controller._tick()
        assert not harness.controller._recovery_manager.in_progress


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

        await harness.controller._tick()
        orch = harness.controller._recovery_manager._orchestrator
        assert orch is not None

        for _ in range(20):
            if orch.phase == RecoveryPhase.DIAGNOSING:
                break
            await harness.controller._tick()
        assert orch.phase == RecoveryPhase.DIAGNOSING

        await harness.controller._tick()
        assert orch.phase == RecoveryPhase.NOTIFY

        await harness.controller._tick()
        assert not harness.controller._recovery_manager.in_progress
        assert harness.notifier is not None
        assert len(harness.notifier.calls) >= 1


# -------------------------------------------------------------------
# Scenario 4: Global timeout
# -------------------------------------------------------------------


class TestGlobalTimeout:
    @pytest.mark.anyio
    async def test_global_timeout_triggers_notify(self) -> None:
        enter_recovery = AlwaysEnterRecoveryDetector(
            trigger=TriggerType.HANG,
            reason="hang detected",
        )
        harness = make_test_controller(
            detectors=[enter_recovery],
            status_sequence=[JobStatus.PENDING] * 100,
        )

        await harness.controller._tick()
        orch = harness.controller._recovery_manager._orchestrator
        assert orch is not None

        from datetime import datetime, timedelta, timezone
        orch._context.recovery_start_time = datetime.now(timezone.utc) - timedelta(seconds=1801)

        await harness.controller._tick()
        assert orch.phase in (RecoveryPhase.NOTIFY, RecoveryPhase.DONE)

        await advance_until_recovery_complete(harness)


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
        assert harness.controller._recovery_manager.in_progress
        initial_detector_count = enter_recovery.call_count

        orch = harness.controller._recovery_manager._orchestrator
        assert orch is not None

        for _ in range(20):
            if orch.phase == RecoveryPhase.MONITORING:
                break
            await harness.controller._tick()
        assert enter_recovery.call_count == initial_detector_count

        active_run_id = harness.controller._rank_roster.run_id
        for i in range(1, 11):
            harness.mini_wandb.log_step(
                run_id=active_run_id, step=i,
                metrics={"iteration": float(i)},
            )
        await harness.controller._tick()

        assert not harness.controller._recovery_manager.in_progress

        harness.controller.rank_roster.rank_placement[0] = "node-0"
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

        await harness.controller._tick()
        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 1.0
        assert get_sample_value(registry, mn.CONTROLLER_RECOVERY_PHASE) is not None
        assert get_sample_value(registry, mn.CONTROLLER_RECOVERY_PHASE) > 0

        orch = harness.controller._recovery_manager._orchestrator
        assert orch is not None

        for _ in range(20):
            if orch.phase == RecoveryPhase.MONITORING:
                break
            await harness.controller._tick()

        active_run_id = harness.controller._rank_roster.run_id
        for i in range(1, 11):
            harness.mini_wandb.log_step(
                run_id=active_run_id, step=i,
                metrics={"iteration": float(i)},
            )

        while harness.controller._recovery_manager.in_progress:
            await harness.controller._tick()

        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 0.0
        assert get_sample_value(registry, mn.CONTROLLER_RECOVERY_PHASE) == 0.0
