"""Tests for dynamic bad-node injection during controller recovery."""

from __future__ import annotations

import pytest
from tests.fast.utils.ft.conftest import FixedDecisionDetector, make_test_controller

from miles.utils.ft.controller.main_state_machine import Recovering
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType


class TestDynamicBadNodeInjection:
    @pytest.mark.anyio
    async def test_dynamic_bad_node_injection(self) -> None:
        """Detector bad nodes are merged into the recovery flow
        and both the initial and injected nodes are evicted."""
        initial_detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-A"],
                reason="initial fault",
                trigger=TriggerType.CRASH,
            )
        )

        hw_detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-B"],
                reason="hw fault during recovery",
                trigger=TriggerType.HARDWARE,
            )
        )

        harness = make_test_controller(
            detectors=[initial_detector, hw_detector],
            register_dummy_rank=False,
        )
        harness.controller._activate_run("test-run")
        harness.controller.rank_roster.rank_placement[0] = "node-A"
        harness.controller.rank_roster.rank_placement[1] = "node-B"

        # Step 1: single tick enters recovery and progresses through the full
        # recovery flow (state machine loops within one tick with instant fakes)
        await harness.controller._tick()
        state = harness.controller._state_machine.state
        assert isinstance(state, Recovering)

        # Step 2: verify both initial and injected nodes were evicted
        assert harness.node_manager.was_ever_marked_bad("node-A")
        assert harness.node_manager.was_ever_marked_bad("node-B")
