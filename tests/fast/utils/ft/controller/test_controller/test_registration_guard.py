from __future__ import annotations

import pytest

from miles.utils.ft.models import ActionType, Decision
from tests.fast.utils.ft.conftest import (
    FixedDecisionDetector,
    make_test_controller,
)


class TestRegistrationGuardNoRanks:
    @pytest.mark.anyio
    async def test_detectors_skipped_when_no_ranks_registered(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=["node-0"],
            reason="should not fire",
        ))
        harness = make_test_controller(
            detectors=[detector],
            registration_grace_ticks=0,
            register_dummy_rank=False,
        )

        await harness.controller._tick()

        assert detector.call_count == 0
        assert not harness.node_manager._bad_nodes

    @pytest.mark.anyio
    async def test_detectors_run_when_ranks_registered(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=["node-0"],
            reason="fault",
        ))
        harness = make_test_controller(
            detectors=[detector],
            registration_grace_ticks=0,
        )
        harness.controller._activate_run("run-1")
        harness.controller._rank_registry.register_training_rank(
            run_id="run-1", rank=0, world_size=1,
            node_id="node-0", exporter_address="http://node-0:9100",
        )

        await harness.controller._tick()

        assert detector.call_count == 1


class TestRegistrationGraceTicks:
    @pytest.mark.anyio
    async def test_detectors_skipped_during_grace_period(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=["node-0"],
            reason="fault",
        ))
        harness = make_test_controller(
            detectors=[detector],
            registration_grace_ticks=3,
        )
        harness.controller._activate_run("run-1")
        harness.controller._rank_registry.register_training_rank(
            run_id="run-1", rank=0, world_size=1,
            node_id="node-0", exporter_address="http://node-0:9100",
        )

        for _ in range(3):
            await harness.controller._tick()

        assert detector.call_count == 0

    @pytest.mark.anyio
    async def test_detectors_run_after_grace_period(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=["node-0"],
            reason="fault",
        ))
        harness = make_test_controller(
            detectors=[detector],
            registration_grace_ticks=2,
        )
        harness.controller._activate_run("run-1")
        harness.controller._rank_registry.register_training_rank(
            run_id="run-1", rank=0, world_size=1,
            node_id="node-0", exporter_address="http://node-0:9100",
        )

        await harness.controller._tick()
        await harness.controller._tick()
        assert detector.call_count == 0

        await harness.controller._tick()
        assert detector.call_count == 1
