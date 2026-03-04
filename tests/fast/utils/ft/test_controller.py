import asyncio

import pytest

from miles.utils.ft.models import ActionType
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import (
    AlwaysMarkBadDetector,
    AlwaysNoneDetector,
    make_test_controller,
)


class TestTickEmptyDetectorChain:
    @pytest.mark.asyncio
    async def test_tick_succeeds_with_no_detectors(self) -> None:
        harness = make_test_controller()
        await harness.controller._tick()
        assert harness.controller._tick_count == 1

    @pytest.mark.asyncio
    async def test_tick_returns_none_decision(self) -> None:
        harness = make_test_controller()
        await harness.controller._tick()
        decision = harness.controller._evaluate_detectors()
        assert decision.action == ActionType.NONE


class TestLogStep:
    @pytest.mark.asyncio
    async def test_log_step_matching_run_id(self) -> None:
        harness = make_test_controller()
        run_id = "run-123"
        await harness.controller.register_rank(
            run_id=run_id, rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        await harness.controller.log_step(
            run_id=run_id, rank=0, step=1,
            metrics={"loss": 3.0, "grad_norm": 1.0},
        )

        assert harness.mini_wandb.latest(metric_name="loss", rank=0) == 3.0

    @pytest.mark.asyncio
    async def test_log_step_mismatched_run_id(self) -> None:
        harness = make_test_controller()
        await harness.controller.register_rank(
            run_id="run-123", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        await harness.controller.log_step(
            run_id="run-OTHER", rank=0, step=1,
            metrics={"loss": 3.0},
        )

        assert harness.mini_wandb.latest(metric_name="loss", rank=0) is None

    @pytest.mark.asyncio
    async def test_log_step_no_active_run(self) -> None:
        harness = make_test_controller()

        await harness.controller.log_step(
            run_id="run-123", rank=0, step=1,
            metrics={"loss": 3.0},
        )

        assert harness.mini_wandb.latest(metric_name="loss", rank=0) == 3.0


class TestRegisterRank:
    @pytest.mark.asyncio
    async def test_new_run_id_updates_state(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        assert harness.controller._active_run_id == "run-1"
        assert harness.controller._rank_placement == {0: "node-0"}

    @pytest.mark.asyncio
    async def test_new_run_id_clears_mini_wandb(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        await harness.controller.log_step(
            run_id="run-1", rank=0, step=1, metrics={"loss": 3.0},
        )
        assert harness.mini_wandb.latest(metric_name="loss", rank=0) == 3.0

        await harness.controller.register_rank(
            run_id="run-2", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        assert harness.mini_wandb.latest(metric_name="loss", rank=0) is None

    @pytest.mark.asyncio
    async def test_same_run_id_different_rank_appends(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        await harness.controller.register_rank(
            run_id="run-1", rank=1, world_size=2,
            node_id="node-1", exporter_address="http://node-1:9090",
        )

        assert harness.controller._rank_placement == {0: "node-0", 1: "node-1"}

    @pytest.mark.asyncio
    async def test_same_run_id_same_rank_updates(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        await harness.controller.log_step(
            run_id="run-1", rank=0, step=1, metrics={"loss": 3.0},
        )

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0-new", exporter_address="http://node-0-new:9090",
        )

        assert harness.controller._rank_placement[0] == "node-0-new"
        assert harness.mini_wandb.latest(metric_name="loss", rank=0) == 3.0

    @pytest.mark.asyncio
    async def test_exporter_address_registered_to_metric_store(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        assert "rank-0" in harness.metric_store._scrape_targets
        assert harness.metric_store._scrape_targets["rank-0"] == "http://node-0:9090"

    @pytest.mark.asyncio
    async def test_new_run_cleans_old_scrape_targets(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        assert "rank-0" in harness.metric_store._scrape_targets

        await harness.controller.register_rank(
            run_id="run-2", rank=1, world_size=2,
            node_id="node-1", exporter_address="http://node-1:9090",
        )

        assert "rank-0" not in harness.metric_store._scrape_targets
        assert "rank-1" in harness.metric_store._scrape_targets


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_stops_run_loop(self) -> None:
        harness = make_test_controller(tick_interval=0.01)

        async def _shutdown_after_delay() -> None:
            await asyncio.sleep(0.05)
            await harness.controller.shutdown()

        task = asyncio.create_task(_shutdown_after_delay())
        await harness.controller.run()
        await task

        assert harness.controller._shutting_down
        assert harness.controller._tick_count >= 1


class TestDetectorChain:
    def test_first_non_none_wins(self) -> None:
        none_detector = AlwaysNoneDetector()
        bad_detector = AlwaysMarkBadDetector()
        harness = make_test_controller(
            detectors=[none_detector, bad_detector],
        )

        decision = harness.controller._evaluate_detectors()
        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert none_detector.call_count == 1
        assert bad_detector.call_count == 1

    def test_short_circuit_after_non_none(self) -> None:
        bad_detector = AlwaysMarkBadDetector()
        trailing_detector = AlwaysNoneDetector()
        harness = make_test_controller(
            detectors=[bad_detector, trailing_detector],
        )

        decision = harness.controller._evaluate_detectors()
        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert bad_detector.call_count == 1
        assert trailing_detector.call_count == 0


class TestTrainingJobStatusInjection:
    @pytest.mark.asyncio
    async def test_tick_injects_training_job_status(self) -> None:
        harness = make_test_controller()

        await harness.controller._tick()

        df = harness.metric_store.instant_query("training_job_status")
        assert not df.is_empty()
        assert df["value"][0] == 1.0

    @pytest.mark.asyncio
    async def test_failed_status_maps_to_negative(self) -> None:
        harness = make_test_controller(
            status_sequence=[JobStatus.FAILED],
        )

        await harness.controller._tick()

        df = harness.metric_store.instant_query("training_job_status")
        assert df["value"][0] == -1.0
