"""Integration tests for FtController: end-to-end data flows."""

import pytest

from tests.fast.utils.ft.conftest import AlwaysNoneDetector, make_test_controller


class TestEmptyDetectorChainMultipleTicks:
    @pytest.mark.anyio
    async def test_three_ticks_succeed(self) -> None:
        harness = make_test_controller()

        for _ in range(3):
            await harness.controller._tick()

        assert harness.controller._tick_count == 3


class TestRegisterRankLogStepQuery:
    @pytest.mark.anyio
    async def test_end_to_end_data_flow(self) -> None:
        harness = make_test_controller()
        run_id = "integ-run-1"

        harness.controller._activate_run(run_id)
        harness.controller.training_rank_roster.register_training_rank(
            run_id=run_id,
            rank=0,
            world_size=4,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1,
        )
        harness.controller.training_rank_roster.register_training_rank(
            run_id=run_id,
            rank=1,
            world_size=4,
            node_id="node-1",
            exporter_address="http://node-1:9090",
            pid=1,
        )

        harness.mini_wandb.log_step(
            run_id=run_id,
            step=1,
            metrics={"loss": 3.5, "grad_norm": 1.2},
        )

        assert harness.mini_wandb.latest(metric_name="loss") == 3.5

        result = harness.mini_wandb.query_last_n_steps(
            metric_name="grad_norm",
            last_n=10,
        )
        assert len(result) == 1
        assert result[0] == (1, 1.2)


class TestRunIdIsolation:
    @pytest.mark.anyio
    async def test_new_run_id_clears_old_data(self) -> None:
        harness = make_test_controller()

        harness.controller._activate_run("run-1")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1,
        )
        harness.mini_wandb.log_step(
            run_id="run-1",
            step=10,
            metrics={"loss": 2.0},
        )
        assert harness.mini_wandb.latest(metric_name="loss") == 2.0

        harness.controller._activate_run("run-2")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-2",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1,
        )

        assert harness.mini_wandb.latest(metric_name="loss") is None
        assert harness.controller._training_rank_roster.run_id == "run-2"
        assert harness.controller._training_rank_roster.rank_placement == {0: "node-0"}

    @pytest.mark.anyio
    async def test_stale_log_step_after_run_switch_is_discarded(self) -> None:
        harness = make_test_controller()

        harness.controller._activate_run("run-1")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1,
        )
        harness.mini_wandb.log_step(
            run_id="run-1",
            step=10,
            metrics={"loss": 2.0},
        )

        harness.controller._activate_run("run-2")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-2",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1,
        )

        harness.mini_wandb.log_step(
            run_id="run-1",
            step=11,
            metrics={"loss": 1.5},
        )

        assert harness.mini_wandb.latest(metric_name="loss") is None


class TestCustomDetectorInTick:
    @pytest.mark.anyio
    async def test_detector_invoked_during_tick(self) -> None:
        detector = AlwaysNoneDetector()
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()

        assert detector.call_count == 1
        assert harness.controller._tick_count == 1
