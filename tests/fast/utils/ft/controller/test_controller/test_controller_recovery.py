"""Tests for dynamic bad-node injection during controller recovery."""

from __future__ import annotations

import pytest
from tests.fast.utils.ft.conftest import AlwaysEnterRecoveryDetector, FixedDecisionDetector, make_test_controller
from tests.fast.utils.ft.utils.controller_fakes import get_training_subsystem_state

from miles.utils.ft.controller.state_machines.subsystem import RecoveringSt
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType


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
        harness.controller.training_rank_roster.rank_placement[0] = "node-A"
        harness.controller.training_rank_roster.rank_placement[1] = "node-B"

        # Step 1: single tick enters recovery and progresses through the full
        # recovery flow (state machine loops within one tick with instant fakes)
        await harness.controller._tick()
        state = get_training_subsystem_state(harness.controller)
        assert isinstance(state, RecoveringSt)

        # Step 2: verify both initial and injected nodes were evicted
        assert harness.node_manager.was_ever_marked_bad("node-A")
        assert harness.node_manager.was_ever_marked_bad("node-B")


class TestRunIdUniqueness:
    @pytest.mark.anyio
    async def test_sequential_recoveries_produce_distinct_run_ids(self) -> None:
        """After two sequential recoveries, the run_id changes each time and all are distinct."""
        harness = make_test_controller(
            detectors=[AlwaysEnterRecoveryDetector()],
            monitoring_success_iterations=0,
        )

        recorded_run_ids: list[str] = []
        original_submit = harness.main_job.submit_job

        async def tracking_submit() -> str:
            run_id = await original_submit()
            recorded_run_ids.append(run_id)
            return run_id

        harness.main_job.submit_job = tracking_submit  # type: ignore[assignment]

        for _ in range(50):
            await harness.controller._tick()
            # Re-register ranks after _activate_run clears the roster
            harness.controller.training_rank_roster.rank_placement[0] = "node-0"
            harness.controller.training_rank_roster.rank_placement[1] = "node-1"
            if len(recorded_run_ids) >= 2:
                break

        assert len(recorded_run_ids) >= 2, f"Expected at least 2 recoveries, got {len(recorded_run_ids)}"
        assert len(set(recorded_run_ids)) == len(recorded_run_ids), f"Duplicate run_ids found: {recorded_run_ids}"
