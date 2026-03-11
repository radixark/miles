from __future__ import annotations

import pytest
from tests.fast.utils.ft.conftest import (
    AlwaysEnterRecoveryDetector,
    FixedDecisionDetector,
    get_sample_value,
    make_test_controller,
    make_test_exporter,
)
from tests.fast.utils.ft.utils.controller_fakes import get_training_subsystem_state

import miles.utils.ft.controller.metrics.metric_names as mn
from miles.utils.ft.controller.state_machines.subsystem import RecoveringSt
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType


class TestEnterRecovery:
    @pytest.mark.anyio
    async def test_creates_recovery_state(self) -> None:
        detector = AlwaysEnterRecoveryDetector()
        harness = make_test_controller(detectors=[detector])
        assert not isinstance(get_training_subsystem_state(harness.controller), RecoveringSt)

        await harness.controller._tick()

        assert isinstance(get_training_subsystem_state(harness.controller), RecoveringSt)

    @pytest.mark.anyio
    async def test_recovery_mode_runs_all_detectors(self) -> None:
        """All detectors run during recovery (is_critical distinction removed).

        Tick 1 enters recovery; collect_evictable_bad_nodes also calls
        detectors during recovery steps, so call_count > 1 after a single tick.
        """
        detector = AlwaysEnterRecoveryDetector()
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()
        assert detector.call_count > 1

    @pytest.mark.anyio
    async def test_detector_finds_bad_nodes_during_recovery(self) -> None:
        """Detectors are called during recovery and their
        findings (bad nodes) are incorporated into the recovery flow."""
        enter_recovery = AlwaysEnterRecoveryDetector()
        hw_detector = FixedDecisionDetector(
            decision=Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-0"],
                reason="hw fault during recovery",
                trigger=TriggerType.HARDWARE,
            )
        )
        harness = make_test_controller(detectors=[enter_recovery, hw_detector])

        await harness.controller._tick()
        assert isinstance(get_training_subsystem_state(harness.controller), RecoveringSt)
        assert hw_detector.call_count > 0
        assert harness.node_manager.was_ever_marked_bad("node-0")

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
