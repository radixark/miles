"""Integration tests: Controller + RecoveryOrchestrator full flow.

Each test drives the Controller through multiple ticks to verify
complete decision → recovery → monitoring lifecycle.
"""
from __future__ import annotations

import pytest

from miles.utils.ft.models import ActionType, Decision, RecoveryPhase
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import (
    AlwaysNoneDetector,
    FixedDecisionDetector,
    get_sample_value,
    inject_gpu_unavailable,
    make_test_controller,
    make_test_exporter,
)


# -------------------------------------------------------------------
# Scenario 1: GPU lost → direct eviction (MARK_BAD_AND_RESTART)
# -------------------------------------------------------------------


class TestGpuLostDirectEviction:
    @pytest.mark.asyncio
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
        assert harness.mini_wandb.latest(metric_name="loss", rank=0) is None


# -------------------------------------------------------------------
# Scenario 2: Training crash → reattempt → success
# -------------------------------------------------------------------


class TestCrashReattemptSuccess:
    @pytest.mark.asyncio
    async def test_crash_reattempt_success(self) -> None:
        enter_recovery = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="training process exited",
        ))
        harness = make_test_controller(
            detectors=[enter_recovery],
            status_sequence=[JobStatus.RUNNING],
        )

        await harness.controller._tick()
        assert harness.controller._recovery_orchestrator is not None
        orch = harness.controller._recovery_orchestrator

        for _ in range(20):
            if orch.phase == RecoveryPhase.MONITORING:
                break
            await harness.controller._tick()
        assert orch.phase == RecoveryPhase.MONITORING

        for i in range(1, 11):
            harness.mini_wandb.log_step(
                run_id="test-run", rank=0, step=i,
                metrics={"iteration": float(i)},
            )

        await harness.controller._tick()
        assert harness.controller._recovery_orchestrator is None


# -------------------------------------------------------------------
# Scenario 3: Training crash → reattempt fails → diagnose → notify
# -------------------------------------------------------------------


class TestCrashReattemptFailDiagnoseNotify:
    @pytest.mark.asyncio
    async def test_crash_diagnose_notify(self) -> None:
        enter_recovery = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="training process exited",
        ))
        harness = make_test_controller(
            detectors=[enter_recovery],
            status_sequence=[JobStatus.FAILED],
        )

        await harness.controller._tick()
        orch = harness.controller._recovery_orchestrator
        assert orch is not None

        for _ in range(20):
            if orch.phase == RecoveryPhase.DIAGNOSING:
                break
            await harness.controller._tick()
        assert orch.phase == RecoveryPhase.DIAGNOSING

        await harness.controller._tick()
        assert orch.phase == RecoveryPhase.NOTIFY

        await harness.controller._tick()
        assert harness.controller._recovery_orchestrator is None
        assert harness.notifier is not None
        assert len(harness.notifier.calls) >= 1


# -------------------------------------------------------------------
# Scenario 4: Global timeout
# -------------------------------------------------------------------


class TestGlobalTimeout:
    @pytest.mark.asyncio
    async def test_global_timeout_triggers_notify(self) -> None:
        enter_recovery = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="hang",
            reason="hang detected",
        ))
        harness = make_test_controller(
            detectors=[enter_recovery],
            status_sequence=[JobStatus.PENDING] * 100,
        )

        await harness.controller._tick()
        orch = harness.controller._recovery_orchestrator
        assert orch is not None

        from datetime import datetime, timedelta, timezone
        orch._context.recovery_start_time = datetime.now(timezone.utc) - timedelta(seconds=1801)

        await harness.controller._tick()
        assert orch.phase in (RecoveryPhase.NOTIFY, RecoveryPhase.DONE)

        for _ in range(5):
            if harness.controller._recovery_orchestrator is None:
                break
            await harness.controller._tick()

        assert harness.controller._recovery_orchestrator is None


# -------------------------------------------------------------------
# Scenario 5: Recovery complete → back to monitoring mode
# -------------------------------------------------------------------


class TestRecoveryCompleteBackToMonitoring:
    @pytest.mark.asyncio
    async def test_back_to_monitoring_after_recovery(self) -> None:
        enter_recovery = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="training process exited",
        ))
        harness = make_test_controller(
            detectors=[enter_recovery],
            status_sequence=[JobStatus.RUNNING],
        )

        await harness.controller._tick()
        assert harness.controller._recovery_orchestrator is not None
        initial_detector_count = enter_recovery.call_count

        orch = harness.controller._recovery_orchestrator
        assert orch is not None

        for _ in range(20):
            if orch.phase == RecoveryPhase.MONITORING:
                break
            await harness.controller._tick()
        assert enter_recovery.call_count == initial_detector_count

        for i in range(1, 11):
            harness.mini_wandb.log_step(
                run_id="test-run", rank=0, step=i,
                metrics={"iteration": float(i)},
            )
        await harness.controller._tick()

        assert harness.controller._recovery_orchestrator is None

        await harness.controller._tick()
        assert enter_recovery.call_count > initial_detector_count


# -------------------------------------------------------------------
# Scenario 6: Exporter mode gauge
# -------------------------------------------------------------------


class TestExporterModeGauge:
    @pytest.mark.asyncio
    async def test_mode_gauge_during_recovery(self) -> None:
        registry, exporter = make_test_exporter()
        enter_recovery = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test",
        ))
        harness = make_test_controller(
            detectors=[enter_recovery],
            status_sequence=[JobStatus.RUNNING],
            controller_exporter=exporter,
        )

        await harness.controller._tick()
        assert get_sample_value(registry, "ft_controller_mode") == 0.0

        await harness.controller._tick()
        assert get_sample_value(registry, "ft_controller_mode") == 1.0
        assert get_sample_value(registry, "ft_controller_recovery_phase") is not None
        assert get_sample_value(registry, "ft_controller_recovery_phase") > 0

        orch = harness.controller._recovery_orchestrator
        assert orch is not None

        for _ in range(20):
            if orch.phase == RecoveryPhase.MONITORING:
                break
            await harness.controller._tick()

        for i in range(1, 11):
            harness.mini_wandb.log_step(
                run_id="test-run", rank=0, step=i,
                metrics={"iteration": float(i)},
            )

        while harness.controller._recovery_orchestrator is not None:
            await harness.controller._tick()

        await harness.controller._tick()
        assert get_sample_value(registry, "ft_controller_mode") == 0.0
        assert get_sample_value(registry, "ft_controller_recovery_phase") == 0.0
