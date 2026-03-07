"""Tests for FtController: submit_initial_training and critical detector edge cases."""
from __future__ import annotations

import pytest

import miles.utils.ft.models.metric_names as mn
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.main_state_machine import DetectingAnomaly, Recovering
from miles.utils.ft.controller.recovery.recovery_stepper import RecoveryDone
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
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
# _collect_critical_bad_nodes — exception isolation
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
        assert isinstance(harness.controller._state_machine.state, Recovering)

        await harness.controller._tick()
        assert crashing_critical.call_count == 1
        assert isinstance(harness.controller._state_machine.state, Recovering)

    @pytest.mark.anyio
    async def test_critical_detector_non_mark_bad_action_is_ignored(self) -> None:
        """A critical detector returning NOTIFY_HUMAN should be silently ignored."""
        enter_recovery = AlwaysEnterRecoveryDetector(reason="test")
        critical_notify = CriticalFixedDecisionDetector(decision=Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="should be ignored during recovery",
            trigger=TriggerType.MISC,
        ))
        harness = make_test_controller(detectors=[enter_recovery, critical_notify])

        await harness.controller._tick()
        state = harness.controller._state_machine.state
        assert isinstance(state, Recovering)

        await harness.controller._tick()
        assert critical_notify.call_count == 1

    @pytest.mark.anyio
    async def test_critical_detector_empty_bad_nodes_is_ignored(self) -> None:
        enter_recovery = AlwaysEnterRecoveryDetector(reason="test")
        critical_empty = CriticalFixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=[],
            reason="no bad nodes",
            trigger=TriggerType.CRASH,
        ))
        harness = make_test_controller(detectors=[enter_recovery, critical_empty])

        await harness.controller._tick()
        assert isinstance(harness.controller._state_machine.state, Recovering)

        await harness.controller._tick()
        assert critical_empty.call_count == 1


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
        assert isinstance(harness.controller._state_machine.state, Recovering)

        from datetime import datetime, timezone
        harness.controller._state_machine._state = Recovering(
            recovery=RecoveryDone(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )

        await harness.controller._tick()
        assert isinstance(harness.controller._state_machine.state, DetectingAnomaly)

        value = get_sample_value(
            registry,
            mn.CONTROLLER_RECOVERY_DURATION_SECONDS + "_count",
        )
        assert value is not None and value >= 1.0
