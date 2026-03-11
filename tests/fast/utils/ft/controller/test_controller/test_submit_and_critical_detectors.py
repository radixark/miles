"""Tests for FtController: submit_initial_job and detector edge cases."""

from __future__ import annotations

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
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.state_machines.subsystem import DetectingAnomaly, Recovering
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType

# ===================================================================
# submit_initial_job
# ===================================================================


class TestSubmitInitialJob:
    @pytest.mark.anyio
    async def test_delegates_to_main_job(self) -> None:
        harness = make_test_controller()

        run_id = await harness.controller.submit_initial_job()

        assert harness.main_job._submitted
        assert isinstance(run_id, str)
        assert len(run_id) > 0


# ===================================================================
# collect_evictable_bad_nodes — exception isolation
# ===================================================================


class _CrashingDetector(BaseFaultDetector):
    """A detector that raises on _evaluate_raw()."""

    def __init__(self) -> None:
        self.call_count = 0

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        self.call_count += 1
        raise RuntimeError("detector internal error")


class TestDetectorExceptionIsolation:
    @pytest.mark.anyio
    async def test_crashing_detector_does_not_break_recovery(self) -> None:
        enter_recovery = AlwaysEnterRecoveryDetector(reason="test")
        crashing = _CrashingDetector()
        harness = make_test_controller(detectors=[enter_recovery, crashing])

        await harness.controller._tick()
        assert isinstance(harness.controller._training_state_machine.state, Recovering)
        assert crashing.call_count > 0

    @pytest.mark.anyio
    async def test_detector_non_mark_bad_action_is_ignored_during_recovery(self) -> None:
        """A detector returning NOTIFY_HUMAN should be silently ignored
        during recovery — only ENTER_RECOVERY with bad_node_ids is acted upon."""
        enter_recovery = AlwaysEnterRecoveryDetector(reason="test")
        notify_detector = FixedDecisionDetector(
            decision=Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason="should be ignored during recovery",
                trigger=TriggerType.MISC,
            )
        )
        harness = make_test_controller(detectors=[enter_recovery, notify_detector])

        await harness.controller._tick()
        state = harness.controller._training_state_machine.state
        assert isinstance(state, Recovering)
        assert notify_detector.call_count > 0

    @pytest.mark.anyio
    async def test_detector_empty_bad_nodes_is_ignored_during_recovery(self) -> None:
        enter_recovery = AlwaysEnterRecoveryDetector(reason="test")
        empty_detector = FixedDecisionDetector(
            decision=Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=[],
                reason="no bad nodes",
                trigger=TriggerType.CRASH,
            )
        )
        harness = make_test_controller(detectors=[enter_recovery, empty_detector])

        await harness.controller._tick()
        assert isinstance(harness.controller._training_state_machine.state, Recovering)
        assert empty_detector.call_count > 0


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

        # Step 1: enter recovery → progresses to MonitoringProgress
        await harness.controller._tick()
        assert isinstance(harness.controller._training_state_machine.state, Recovering)

        # Step 2: inject training progress so MonitoringProgress succeeds
        run_id = harness.controller.training_rank_roster.run_id
        harness.mini_wandb.log_step(run_id=run_id, step=10, metrics={"iteration": 10})

        # Step 3: tick completes recovery → back to DetectingAnomaly
        await harness.controller._tick()
        assert isinstance(harness.controller._training_state_machine.state, DetectingAnomaly)

        value = get_sample_value(
            registry,
            mn.CONTROLLER_RECOVERY_DURATION_SECONDS + "_count",
        )
        assert value is not None and value >= 1.0
