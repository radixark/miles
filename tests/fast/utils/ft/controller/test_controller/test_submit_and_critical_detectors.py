"""Tests for FtController: submit_initial_training and critical detector edge cases."""
from __future__ import annotations

import pytest

import miles.utils.ft.models.metric_names as mn
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.models.fault import ActionType, Decision
from miles.utils.ft.models.recovery import RecoveryPhase
from miles.utils.ft.protocols.platform import JobStatus
from tests.fast.utils.ft.conftest import (
    AlwaysEnterRecoveryDetector,
    CriticalFixedDecisionDetector,
    get_sample_value,
    make_test_controller,
    make_test_exporter,
)


# ===================================================================
# submit_initial_training
# ===================================================================


class TestSubmitInitialTraining:
    @pytest.mark.anyio
    async def test_delegates_to_training_job(self) -> None:
        harness = make_test_controller()

        run_id = await harness.controller.submit_initial_training()

        assert harness.training_job._submitted
        assert isinstance(run_id, str)
        assert len(run_id) > 0


# ===================================================================
# _run_critical_detectors_during_recovery — exception isolation
# ===================================================================


class _CrashingCriticalDetector(BaseFaultDetector):
    """A critical detector that raises on evaluate()."""

    is_critical = True

    def __init__(self) -> None:
        self.call_count = 0

    def evaluate(self, ctx: DetectorContext) -> Decision:
        self.call_count += 1
        raise RuntimeError("critical detector internal error")


class TestCriticalDetectorExceptionIsolation:
    @pytest.mark.anyio
    async def test_crashing_critical_detector_does_not_break_recovery(self) -> None:
        enter_recovery = AlwaysEnterRecoveryDetector(reason="test")
        crashing_critical = _CrashingCriticalDetector()
        harness = make_test_controller(detectors=[enter_recovery, crashing_critical])

        await harness.controller._tick()
        assert harness.controller._recovery_manager.in_progress

        await harness.controller._tick()
        assert crashing_critical.call_count == 1
        assert harness.controller._recovery_manager.in_progress

    @pytest.mark.anyio
    async def test_critical_detector_non_mark_bad_action_is_ignored(self) -> None:
        """A critical detector returning NOTIFY_HUMAN should be silently ignored."""
        enter_recovery = AlwaysEnterRecoveryDetector(reason="test")
        critical_notify = CriticalFixedDecisionDetector(decision=Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="should be ignored during recovery",
        ))
        harness = make_test_controller(detectors=[enter_recovery, critical_notify])

        await harness.controller._tick()
        orch = harness.controller._recovery_manager._orchestrator
        assert orch is not None
        bad_before = list(orch.bad_node_ids)

        await harness.controller._tick()
        assert critical_notify.call_count == 1
        assert orch.bad_node_ids == bad_before

    @pytest.mark.anyio
    async def test_critical_detector_empty_bad_nodes_is_ignored(self) -> None:
        enter_recovery = AlwaysEnterRecoveryDetector(reason="test")
        critical_empty = CriticalFixedDecisionDetector(decision=Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=[],
            reason="no bad nodes",
        ))
        harness = make_test_controller(detectors=[enter_recovery, critical_empty])

        await harness.controller._tick()
        orch = harness.controller._recovery_manager._orchestrator
        assert orch is not None

        await harness.controller._tick()
        assert critical_empty.call_count == 1
        assert orch.bad_node_ids == []


# ===================================================================
# Recovery completion — exporter metrics
# ===================================================================


class TestRecoveryCompletionMetrics:
    @pytest.mark.anyio
    async def test_observe_recovery_duration_called_on_completion(self) -> None:
        registry, exporter = make_test_exporter()
        detector = AlwaysEnterRecoveryDetector(reason="test")
        harness = make_test_controller(
            detectors=[detector],
            controller_exporter=exporter,
            status_sequence=[JobStatus.RUNNING],
        )

        await harness.controller._tick()
        assert harness.controller._recovery_manager.in_progress

        harness.controller._recovery_manager._orchestrator._context.phase = RecoveryPhase.DONE

        await harness.controller._tick()
        assert not harness.controller._recovery_manager.in_progress

        value = get_sample_value(
            registry,
            mn.CONTROLLER_RECOVERY_DURATION_SECONDS + "_count",
        )
        assert value is not None and value >= 1.0

    @pytest.mark.anyio
    async def test_phase_history_preserved_after_recovery_complete(self) -> None:
        detector = AlwaysEnterRecoveryDetector(reason="test")
        harness = make_test_controller(
            detectors=[detector],
            status_sequence=[JobStatus.RUNNING],
        )

        await harness.controller._tick()
        orch = harness.controller._recovery_manager._orchestrator
        assert orch is not None
        orch._context.phase = RecoveryPhase.DONE

        await harness.controller._tick()
        assert not harness.controller._recovery_manager.in_progress

        status = harness.controller.get_status()
        assert status.phase_history is not None
        assert len(status.phase_history) >= 1
