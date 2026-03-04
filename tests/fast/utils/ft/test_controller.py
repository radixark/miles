import asyncio

import pytest

from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import make_test_controller


class _AlwaysNoneDetector(BaseFaultDetector):
    def __init__(self) -> None:
        self.call_count = 0

    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
    ) -> Decision:
        self.call_count += 1
        return Decision(action=ActionType.NONE, reason="always none")


class _AlwaysMarkBadDetector(BaseFaultDetector):
    def __init__(self) -> None:
        self.call_count = 0

    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
    ) -> Decision:
        self.call_count += 1
        return Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=["node-1"],
            reason="test fault detected",
        )


class TestTickEmptyDetectorChain:
    @pytest.mark.asyncio
    async def test_tick_succeeds_with_no_detectors(self) -> None:
        controller, _, _, _, _ = make_test_controller()
        await controller._tick()
        assert controller._tick_count == 1

    @pytest.mark.asyncio
    async def test_tick_returns_none_decision(self) -> None:
        controller, _, _, _, _ = make_test_controller()
        await controller._tick()
        decision = controller._evaluate_detectors()
        assert decision.action == ActionType.NONE


class TestLogStep:
    @pytest.mark.asyncio
    async def test_log_step_matching_run_id(self) -> None:
        controller, _, _, _, mini_wandb = make_test_controller()
        run_id = "run-123"
        await controller.register_rank(
            run_id=run_id, rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        await controller.log_step(
            run_id=run_id, rank=0, step=1,
            metrics={"loss": 3.0, "grad_norm": 1.0},
        )

        assert mini_wandb.latest(metric_name="loss", rank=0) == 3.0

    @pytest.mark.asyncio
    async def test_log_step_mismatched_run_id(self) -> None:
        controller, _, _, _, mini_wandb = make_test_controller()
        await controller.register_rank(
            run_id="run-123", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        await controller.log_step(
            run_id="run-OTHER", rank=0, step=1,
            metrics={"loss": 3.0},
        )

        assert mini_wandb.latest(metric_name="loss", rank=0) is None

    @pytest.mark.asyncio
    async def test_log_step_no_active_run(self) -> None:
        controller, _, _, _, mini_wandb = make_test_controller()

        await controller.log_step(
            run_id="run-123", rank=0, step=1,
            metrics={"loss": 3.0},
        )

        assert mini_wandb.latest(metric_name="loss", rank=0) == 3.0


class TestRegisterRank:
    @pytest.mark.asyncio
    async def test_new_run_id_updates_state(self) -> None:
        controller, _, _, _, mini_wandb = make_test_controller()

        await controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        assert controller._active_run_id == "run-1"
        assert controller._rank_placement == {0: "node-0"}

    @pytest.mark.asyncio
    async def test_new_run_id_clears_mini_wandb(self) -> None:
        controller, _, _, _, mini_wandb = make_test_controller()

        await controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        await controller.log_step(
            run_id="run-1", rank=0, step=1, metrics={"loss": 3.0},
        )
        assert mini_wandb.latest(metric_name="loss", rank=0) == 3.0

        await controller.register_rank(
            run_id="run-2", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        assert mini_wandb.latest(metric_name="loss", rank=0) is None

    @pytest.mark.asyncio
    async def test_same_run_id_different_rank_appends(self) -> None:
        controller, _, _, _, _ = make_test_controller()

        await controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        await controller.register_rank(
            run_id="run-1", rank=1, world_size=2,
            node_id="node-1", exporter_address="http://node-1:9090",
        )

        assert controller._rank_placement == {0: "node-0", 1: "node-1"}

    @pytest.mark.asyncio
    async def test_same_run_id_same_rank_updates(self) -> None:
        controller, _, _, _, mini_wandb = make_test_controller()

        await controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        await controller.log_step(
            run_id="run-1", rank=0, step=1, metrics={"loss": 3.0},
        )

        await controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0-new", exporter_address="http://node-0-new:9090",
        )

        assert controller._rank_placement[0] == "node-0-new"
        assert mini_wandb.latest(metric_name="loss", rank=0) == 3.0

    @pytest.mark.asyncio
    async def test_exporter_address_registered_to_metric_store(self) -> None:
        controller, _, _, metric_store, _ = make_test_controller()

        await controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        assert "rank-0" in metric_store._scrape_targets
        assert metric_store._scrape_targets["rank-0"] == "http://node-0:9090"


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_stops_run_loop(self) -> None:
        controller, _, _, _, _ = make_test_controller(tick_interval=0.01)

        async def _shutdown_after_delay() -> None:
            await asyncio.sleep(0.05)
            await controller.shutdown()

        task = asyncio.create_task(_shutdown_after_delay())
        await controller.run()
        await task

        assert controller._shutting_down
        assert controller._tick_count >= 1


class TestDetectorChain:
    @pytest.mark.asyncio
    async def test_first_non_none_wins(self) -> None:
        none_detector = _AlwaysNoneDetector()
        bad_detector = _AlwaysMarkBadDetector()
        controller, _, _, _, _ = make_test_controller(
            detectors=[none_detector, bad_detector],
        )

        decision = controller._evaluate_detectors()
        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert none_detector.call_count == 1
        assert bad_detector.call_count == 1

    @pytest.mark.asyncio
    async def test_short_circuit_after_non_none(self) -> None:
        bad_detector = _AlwaysMarkBadDetector()
        trailing_detector = _AlwaysNoneDetector()
        controller, _, _, _, _ = make_test_controller(
            detectors=[bad_detector, trailing_detector],
        )

        decision = controller._evaluate_detectors()
        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert bad_detector.call_count == 1
        assert trailing_detector.call_count == 0


class TestTrainingJobStatusInjection:
    @pytest.mark.asyncio
    async def test_tick_injects_training_job_status(self) -> None:
        controller, _, _, metric_store, _ = make_test_controller()

        await controller._tick()

        df = metric_store.instant_query("training_job_status")
        assert not df.is_empty()
        assert df["value"][0] == 1.0

    @pytest.mark.asyncio
    async def test_failed_status_maps_to_negative(self) -> None:
        controller, _, _, metric_store, _ = make_test_controller(
            status_sequence=[JobStatus.FAILED],
        )

        await controller._tick()

        df = metric_store.instant_query("training_job_status")
        assert df["value"][0] == -1.0
