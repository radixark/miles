from __future__ import annotations

import pytest
from tests.fast.utils.ft.conftest import AlwaysNoneDetector, make_test_controller

from miles.utils.ft.controller.runtime_config import ControllerRuntimeConfig


class TestRegistrationGuardNoRanks:
    @pytest.mark.anyio
    async def test_detectors_skipped_when_no_ranks_registered(self) -> None:
        detector = AlwaysNoneDetector()
        harness = make_test_controller(
            detectors=[detector],
            register_dummy_rank=False,
        )

        await harness.controller._tick()

        assert detector.call_count == 0
        assert not harness.node_manager._bad_nodes

    @pytest.mark.anyio
    async def test_detectors_run_when_ranks_registered(self) -> None:
        detector = AlwaysNoneDetector()
        harness = make_test_controller(
            detectors=[detector],
        )
        harness.controller._activate_run("run-1")
        harness.subsystem_hub.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=1,
            node_id="node-0",
            exporter_address="http://node-0:9100",
            pid=1,
        )

        await harness.controller._tick()

        assert detector.call_count == 1


class TestRegistrationGraceTicks:
    @pytest.mark.anyio
    async def test_detectors_skipped_during_grace_period(self) -> None:
        detector = AlwaysNoneDetector()
        harness = make_test_controller(
            detectors=[detector],
            runtime_config=ControllerRuntimeConfig(
                tick_interval=0.01,
                registration_grace_ticks=3,
            ),
        )
        harness.controller._activate_run("run-1")
        harness.subsystem_hub.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=1,
            node_id="node-0",
            exporter_address="http://node-0:9100",
            pid=1,
        )

        for _ in range(3):
            await harness.controller._tick()

        assert detector.call_count == 0

    @pytest.mark.anyio
    async def test_detectors_run_after_grace_period(self) -> None:
        detector = AlwaysNoneDetector()
        harness = make_test_controller(
            detectors=[detector],
            runtime_config=ControllerRuntimeConfig(
                tick_interval=0.01,
                registration_grace_ticks=2,
            ),
        )
        harness.controller._activate_run("run-1")
        harness.subsystem_hub.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=1,
            node_id="node-0",
            exporter_address="http://node-0:9100",
            pid=1,
        )

        await harness.controller._tick()
        await harness.controller._tick()
        assert detector.call_count == 0

        await harness.controller._tick()
        assert detector.call_count == 1
