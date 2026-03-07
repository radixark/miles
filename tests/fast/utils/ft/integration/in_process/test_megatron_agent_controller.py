"""Integration tests for FtMegatronAgent + FtController data flows.

These tests directly instantiate both components (bypassing Ray) to verify
the heartbeat data path: agent step() → exporter gauges → MiniPrometheus scrape,
and the controller's log_step() → MiniWandb store.

Training metrics are forwarded by FtTrackingAgent (via tracking_utils.log()),
not by FtMegatronAgent.step().
"""

import httpx
import pytest

from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent

from tests.fast.utils.ft.conftest import make_test_controller


def _make_agent(rank: int = 0, world_size: int = 4) -> FtTrainingRankAgent:
    return FtTrainingRankAgent(rank=rank, world_size=world_size)


class TestStepToLogStepFlow:
    @pytest.mark.anyio
    async def test_step_metrics_arrive_in_mini_wandb(self) -> None:
        harness = make_test_controller()
        run_id = "integ-megatron-1"

        harness.controller._activate_run(run_id)
        harness.controller.rank_roster.register_training_rank(
            run_id=run_id, rank=0, world_size=4,
            node_id="node-0", exporter_address="http://localhost:9999",
            pid=1,
        )

        harness.mini_wandb.log_step(
            run_id=run_id, step=5,
            metrics={"loss": 2.5, "grad_norm": 1.1},
        )

        assert harness.mini_wandb.latest(metric_name="loss") == 2.5
        assert harness.mini_wandb.latest(metric_name="grad_norm") == 1.1


class TestRegisterRankPlacement:
    @pytest.mark.anyio
    async def test_register_rank_records_placement(self) -> None:
        harness = make_test_controller()
        run_id = "integ-megatron-2"

        harness.controller._activate_run(run_id)
        harness.controller.rank_roster.register_training_rank(
            run_id=run_id, rank=0, world_size=4,
            node_id="node-0", exporter_address="http://node-0:9090",
            pid=1,
        )
        harness.controller.rank_roster.register_training_rank(
            run_id=run_id, rank=1, world_size=4,
            node_id="node-1", exporter_address="http://node-1:9090",
            pid=1,
        )

        assert harness.controller._rank_roster.rank_placement == {0: "node-0", 1: "node-1"}


class TestScrapeTargetRegistration:
    @pytest.mark.anyio
    async def test_register_rank_adds_scrape_target(self) -> None:
        harness = make_test_controller()
        agent = _make_agent(rank=0, world_size=4)
        try:
            run_id = "integ-megatron-3"
            exporter_address = agent.get_exporter_address()

            harness.controller._activate_run(run_id)
            harness.controller.rank_roster.register_training_rank(
                run_id=run_id, rank=0, world_size=4,
                node_id="node-0", exporter_address=exporter_address,
                pid=1,
            )

            assert "rank-0" in harness.metric_store._scrape_targets
        finally:
            agent.shutdown()


class TestHeartbeatScrape:
    @pytest.mark.anyio
    async def test_scrape_reads_heartbeat_gauges(self) -> None:
        harness = make_test_controller()
        agent = _make_agent(rank=0, world_size=4)
        try:
            run_id = "integ-megatron-4"
            exporter_address = agent.get_exporter_address()

            harness.controller._activate_run(run_id)
            harness.controller.rank_roster.register_training_rank(
                run_id=run_id, rank=0, world_size=4,
                node_id="node-0", exporter_address=exporter_address,
                pid=1,
            )

            agent.set_phase("training")
            agent.step()

            await harness.metric_store.scrape_once()

            result = harness.metric_store.query_latest("miles_ft_agent_heartbeat")
            assert len(result) > 0
            row = result.row(0, named=True)
            assert row["value"] >= 1.0
        finally:
            agent.shutdown()


class TestRunIdClear:
    @pytest.mark.anyio
    async def test_new_run_id_clears_mini_wandb(self) -> None:
        harness = make_test_controller()
        run_id_1 = "integ-megatron-run-1"
        run_id_2 = "integ-megatron-run-2"

        harness.controller._activate_run(run_id_1)
        harness.controller.rank_roster.register_training_rank(
            run_id=run_id_1, rank=0, world_size=2,
            node_id="node-0", exporter_address="http://localhost:9999",
            pid=1,
        )
        harness.mini_wandb.log_step(
            run_id=run_id_1, step=10,
            metrics={"loss": 2.0},
        )
        assert harness.mini_wandb.latest(metric_name="loss") == 2.0

        harness.controller._activate_run(run_id_2)
        harness.controller.rank_roster.register_training_rank(
            run_id=run_id_2, rank=0, world_size=2,
            node_id="node-0", exporter_address="http://localhost:9999",
            pid=1,
        )

        assert harness.mini_wandb.latest(metric_name="loss") is None


class TestControllerUnreachable:
    def test_step_without_controller_does_not_raise(self) -> None:
        agent = _make_agent(rank=0, world_size=4)
        try:
            agent.step()
        finally:
            agent.shutdown()


class TestPhaseSwitch:
    @pytest.mark.anyio
    async def test_phase_switch_visible_in_exporter(self) -> None:
        agent = _make_agent(rank=0, world_size=4)
        try:
            agent.set_phase("training")
            agent.step()
            address = agent.get_exporter_address()
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{address}/metrics")
            assert "1.0" in response.text

            agent.set_phase("checkpoint_saving")
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{address}/metrics")
            assert "2.0" in response.text

            agent.set_phase("training")
            agent.step()
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{address}/metrics")
            text = response.text
            assert 'miles_ft_training_phase{node_id=' in text
        finally:
            agent.shutdown()
