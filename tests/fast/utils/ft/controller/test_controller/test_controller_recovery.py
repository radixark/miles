"""Tests for dynamic bad-node injection during controller recovery."""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.main_state_machine import DetectingAnomaly, Recovering
from miles.utils.ft.controller.recovery.recovery_stepper import RealtimeChecks
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from tests.fast.utils.ft.conftest import (
    CriticalFixedDecisionDetector,
    FixedDecisionDetector,
    make_test_controller,
)


class TestDynamicBadNodeInjection:
    @pytest.mark.anyio
    async def test_dynamic_bad_node_injection(self) -> None:
        """Start recovery with initial bad nodes, inject additional bad nodes
        via critical detector during recovery, verify they're merged."""
        initial_detector = FixedDecisionDetector(Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=["node-A"],
            reason="initial fault",
            trigger=TriggerType.CRASH,
        ))

        critical = CriticalFixedDecisionDetector(Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=["node-B"],
            reason="critical fault during recovery",
            trigger=TriggerType.HARDWARE,
        ))

        harness = make_test_controller(detectors=[initial_detector, critical])

        # Step 1: Enter recovery with initial bad nodes
        await harness.controller._tick()
        state = harness.controller._state_machine.state
        assert isinstance(state, Recovering)
        assert isinstance(state.recovery, RealtimeChecks)
        assert "node-A" in state.recovery.pre_identified_bad_nodes

        # Step 2: Critical detector injects new bad nodes during recovery
        await harness.controller._tick()
        state = harness.controller._state_machine.state
        assert isinstance(state, Recovering)
        assert isinstance(state.recovery, RealtimeChecks)
        assert set(state.recovery.pre_identified_bad_nodes) >= {"node-A", "node-B"}
