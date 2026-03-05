"""Integration tests for FtController: end-to-end data flows."""

import pytest

from miles.utils.ft.models import ActionType
from tests.fast.utils.ft.helpers import AlwaysMarkBadDetector, make_test_controller


class TestEmptyDetectorChainMultipleTicks:
    @pytest.mark.asyncio
    async def test_three_ticks_succeed(self) -> None:
        harness = make_test_controller()

        for _ in range(3):
            await harness.controller._tick()

        assert harness.controller._tick_count == 3


class TestRegisterRankLogStepQuery:
    @pytest.mark.asyncio
    async def test_end_to_end_data_flow(self) -> None:
        harness = make_test_controller()
        run_id = "integ-run-1"

        await harness.controller.register_rank(
            run_id=run_id, rank=0, world_size=4,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        await harness.controller.register_rank(
            run_id=run_id, rank=1, world_size=4,
            node_id="node-1", exporter_address="http://node-1:9090",
        )

        await harness.controller.log_step(
            run_id=run_id, rank=0, step=1,
            metrics={"loss": 3.5, "grad_norm": 1.2},
        )
        await harness.controller.log_step(
            run_id=run_id, rank=1, step=1,
            metrics={"loss": 3.4, "grad_norm": 1.1},
        )

        assert harness.mini_wandb.latest(metric_name="loss", rank=0) == 3.5
        assert harness.mini_wandb.latest(metric_name="loss", rank=1) == 3.4

        result_rank0 = harness.mini_wandb.query_last_n_steps(
            metric_name="grad_norm", rank=0, last_n=10,
        )
        assert len(result_rank0) == 1
        assert result_rank0[0] == (1, 1.2)


class TestRunIdIsolation:
    @pytest.mark.asyncio
    async def test_new_run_id_clears_old_data(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        await harness.controller.log_step(
            run_id="run-1", rank=0, step=10,
            metrics={"loss": 2.0},
        )
        assert harness.mini_wandb.latest(metric_name="loss", rank=0) == 2.0

        await harness.controller.register_rank(
            run_id="run-2", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        assert harness.mini_wandb.latest(metric_name="loss", rank=0) is None
        assert harness.controller._rank_registry.active_run_id == "run-2"
        assert harness.controller._rank_registry.rank_placement == {0: "node-0"}

    @pytest.mark.asyncio
    async def test_stale_log_step_after_run_switch_is_discarded(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        await harness.controller.log_step(
            run_id="run-1", rank=0, step=10,
            metrics={"loss": 2.0},
        )

        await harness.controller.register_rank(
            run_id="run-2", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        await harness.controller.log_step(
            run_id="run-1", rank=0, step=11,
            metrics={"loss": 1.5},
        )

        assert harness.mini_wandb.latest(metric_name="loss", rank=0) is None


class TestCustomDetectorInTick:
    @pytest.mark.asyncio
    async def test_detector_invoked_during_tick(self) -> None:
        detector = AlwaysMarkBadDetector()
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()

        assert detector.call_count == 1
        assert harness.controller._tick_count == 1
