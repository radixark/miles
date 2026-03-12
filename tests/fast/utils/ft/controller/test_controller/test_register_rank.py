from __future__ import annotations

import logging

import pytest

from tests.fast.utils.ft.conftest import make_test_controller


class TestRegisterRank:
    @pytest.mark.anyio
    async def test_new_run_id_updates_state(self) -> None:
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

        assert harness.controller.training_rank_roster.run_id == "run-1"
        assert harness.controller.training_rank_roster.rank_placement == {0: "node-0"}

    @pytest.mark.anyio
    async def test_same_run_id_different_rank_appends(self) -> None:
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
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=1,
            world_size=2,
            node_id="node-1",
            exporter_address="http://node-1:9090",
            pid=1,
        )

        assert harness.controller.training_rank_roster.rank_placement == {0: "node-0", 1: "node-1"}

    @pytest.mark.anyio
    async def test_same_run_id_same_rank_updates(self) -> None:
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
            step=1,
            metrics={"loss": 3.0},
        )

        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=2,
            node_id="node-0-new",
            exporter_address="http://node-0-new:9090",
            pid=1,
        )

        assert harness.controller.training_rank_roster.rank_placement[0] == "node-0-new"
        assert harness.mini_wandb.latest(metric_name="loss") == 3.0

    @pytest.mark.anyio
    async def test_exporter_address_registered_to_metric_store(self) -> None:
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

        assert "rank-0" in harness.metric_store._scrape_targets
        assert harness.metric_store._scrape_targets["rank-0"] == "http://node-0:9090"

    @pytest.mark.anyio
    async def test_new_run_cleans_old_scrape_targets(self) -> None:
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
        assert "rank-0" in harness.metric_store._scrape_targets

        harness.controller._activate_run("run-2")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-2",
            rank=1,
            world_size=2,
            node_id="node-1",
            exporter_address="http://node-1:9090",
            pid=1,
        )

        assert "rank-0" not in harness.metric_store._scrape_targets
        assert "rank-1" in harness.metric_store._scrape_targets

    @pytest.mark.anyio
    async def test_new_run_cleans_multiple_old_scrape_targets(self) -> None:
        harness = make_test_controller()

        harness.controller._activate_run("run-1")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=4,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1,
        )
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=1,
            world_size=4,
            node_id="node-1",
            exporter_address="http://node-1:9090",
            pid=1,
        )
        assert "rank-0" in harness.metric_store._scrape_targets
        assert "rank-1" in harness.metric_store._scrape_targets

        harness.controller._activate_run("run-2")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-2",
            rank=2,
            world_size=4,
            node_id="node-2",
            exporter_address="http://node-2:9090",
            pid=1,
        )

        assert "rank-0" not in harness.metric_store._scrape_targets
        assert "rank-1" not in harness.metric_store._scrape_targets
        assert "rank-2" in harness.metric_store._scrape_targets

    @pytest.mark.anyio
    async def test_partial_registration_tick_still_runs(self, caplog: pytest.LogCaptureFixture) -> None:
        """world_size=4 but only 3 ranks register; tick runs normally
        but emits a WARNING about incomplete registration.
        """
        harness = make_test_controller()

        harness.controller._activate_run("run-1")
        for rank in range(3):
            harness.controller.training_rank_roster.register_training_rank(
                run_id="run-1",
                rank=rank,
                world_size=4,
                node_id=f"node-{rank}",
                exporter_address=f"http://node-{rank}:9090",
                pid=1,
            )

        assert len(harness.controller.training_rank_roster.rank_placement) == 3
        assert 3 not in harness.controller.training_rank_roster.rank_placement
        assert harness.controller.training_rank_roster.expected_world_size == 4

        with caplog.at_level(logging.WARNING):
            await harness.controller._tick()

        assert harness.controller._tick_count == 1
        assert "incomplete_rank_registration" in caplog.text
        assert "registered=3" in caplog.text
        assert "expected=4" in caplog.text

    @pytest.mark.anyio
    async def test_full_registration_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """All 4/4 ranks registered — tick should not emit WARNING."""
        harness = make_test_controller()

        harness.controller._activate_run("run-1")
        for rank in range(4):
            harness.controller.training_rank_roster.register_training_rank(
                run_id="run-1",
                rank=rank,
                world_size=4,
                node_id=f"node-{rank}",
                exporter_address=f"http://node-{rank}:9090",
                pid=1,
            )

        assert harness.controller.training_rank_roster.expected_world_size == 4
        assert len(harness.controller.training_rank_roster.rank_placement) == 4

        with caplog.at_level(logging.WARNING):
            await harness.controller._tick()

        assert "incomplete_rank_registration" not in caplog.text

    @pytest.mark.anyio
    async def test_expected_world_size_reset_on_new_run(self) -> None:
        """When a new run_id arrives, _expected_world_size is reset."""
        harness = make_test_controller()

        harness.controller._activate_run("run-1")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=8,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1,
        )
        assert harness.controller.training_rank_roster.expected_world_size == 8

        harness.controller._activate_run("run-2")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-2",
            rank=0,
            world_size=4,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1,
        )
        assert harness.controller.training_rank_roster.expected_world_size == 4
        assert harness.controller.training_rank_roster.rank_placement == {0: "node-0"}

    @pytest.mark.anyio
    async def test_register_rank_stores_pid(self) -> None:
        harness = make_test_controller()

        harness.controller._activate_run("run-1")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1234,
        )

        assert harness.controller.training_rank_roster.rank_pids == {0: 1234}

    @pytest.mark.anyio
    async def test_new_run_id_clears_rank_pids(self) -> None:
        harness = make_test_controller()

        harness.controller._activate_run("run-1")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1234,
        )
        assert harness.controller.training_rank_roster.rank_pids == {0: 1234}

        harness.controller._activate_run("run-2")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-2",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=5678,
        )
        assert harness.controller.training_rank_roster.rank_pids == {0: 5678}


class TestGetRankPidsForNode:
    @pytest.mark.anyio
    async def test_returns_pids_for_matching_node(self) -> None:
        harness = make_test_controller()

        harness.controller._activate_run("run-1")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=4,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=100,
        )
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=1,
            world_size=4,
            node_id="node-0",
            exporter_address="http://node-0:9091",
            pid=101,
        )
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=2,
            world_size=4,
            node_id="node-1",
            exporter_address="http://node-1:9090",
            pid=200,
        )

        result = harness.controller.training_rank_roster.get_rank_pids_for_node("node-0")
        assert result == {0: 100, 1: 101}

    @pytest.mark.anyio
    async def test_returns_empty_for_unknown_node(self) -> None:
        harness = make_test_controller()

        harness.controller._activate_run("run-1")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=100,
        )

        result = harness.controller.training_rank_roster.get_rank_pids_for_node("node-999")
        assert result == {}


class TestLogStep:
    @pytest.mark.anyio
    async def test_log_step_matching_run_id(self) -> None:
        harness = make_test_controller()
        run_id = "run-123"
        harness.controller._activate_run(run_id)
        harness.controller.training_rank_roster.register_training_rank(
            run_id=run_id,
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1,
        )

        harness.mini_wandb.log_step(
            run_id=run_id,
            step=1,
            metrics={"loss": 3.0, "grad_norm": 1.0},
        )

        assert harness.mini_wandb.latest(metric_name="loss") == 3.0

    @pytest.mark.anyio
    async def test_log_step_mismatched_run_id(self) -> None:
        harness = make_test_controller()
        harness.controller._activate_run("run-123")
        harness.controller.training_rank_roster.register_training_rank(
            run_id="run-123",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=1,
        )

        harness.mini_wandb.log_step(
            run_id="run-OTHER",
            step=1,
            metrics={"loss": 3.0},
        )

        assert harness.mini_wandb.latest(metric_name="loss") is None

    @pytest.mark.anyio
    async def test_log_step_no_active_run(self) -> None:
        harness = make_test_controller(register_dummy_rank=False)

        harness.mini_wandb.log_step(
            run_id="run-123",
            step=1,
            metrics={"loss": 3.0},
        )

        assert harness.hub.training_rank_roster_box.value is None
        assert harness.mini_wandb.latest(metric_name="loss") is None
        harness.mini_wandb.set_active_run_id("run-123")
        assert harness.mini_wandb.latest(metric_name="loss") == 3.0
