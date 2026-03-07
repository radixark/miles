from __future__ import annotations

import pytest

import miles.utils.ft.models.metric_names as mn
from miles.utils.ft.controller.main_state_machine import DetectingAnomaly, Recovering
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.protocols.platform import JobStatus
from tests.fast.utils.ft.conftest import (
    AlwaysEnterRecoveryDetector,
    CriticalFixedDecisionDetector,
    get_sample_value,
    make_test_controller,
    make_test_exporter,
)


class TestEnterRecovery:
    @pytest.mark.anyio
    async def test_creates_recovery_state(self) -> None:
        detector = AlwaysEnterRecoveryDetector()
        harness = make_test_controller(detectors=[detector])
        assert not isinstance(harness.controller._state_machine.state, Recovering)

        await harness.controller._tick()

        assert isinstance(harness.controller._state_machine.state, Recovering)

    @pytest.mark.anyio
    async def test_recovery_mode_skips_non_critical_detectors(self) -> None:
        detector = AlwaysEnterRecoveryDetector()
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()
        initial_count = detector.call_count

        await harness.controller._tick()
        assert detector.call_count == initial_count

    @pytest.mark.anyio
    async def test_critical_detector_runs_during_recovery(self) -> None:
        enter_recovery = AlwaysEnterRecoveryDetector()
        critical = CriticalFixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=["node-new-bad"],
            reason="critical hw fault during recovery",
            trigger=TriggerType.HARDWARE,
        ))
        harness = make_test_controller(detectors=[enter_recovery, critical])

        await harness.controller._tick()
        assert isinstance(harness.controller._state_machine.state, Recovering)
        assert critical.call_count == 0

        await harness.controller._tick()
        assert critical.call_count == 1

        state = harness.controller._state_machine.state
        assert isinstance(state, Recovering)
        from miles.utils.ft.controller.recovery.recovery_stepper import RealtimeChecks
        assert isinstance(state.recovery, RealtimeChecks)
        assert "node-new-bad" in state.recovery.pre_identified_bad_nodes

    @pytest.mark.anyio
    async def test_exporter_mode_reflects_recovery(self) -> None:
        registry, exporter = make_test_exporter()
        detector = AlwaysEnterRecoveryDetector()
        harness = make_test_controller(
            detectors=[detector],
            controller_exporter=exporter,
        )

        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 0.0

        await harness.controller._tick()

        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 1.0
