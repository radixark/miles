from __future__ import annotations

import asyncio
import logging

import pytest
from prometheus_client import CollectorRegistry

from miles.utils.ft.controller.controller_exporter import ControllerExporter
from miles.utils.ft.models import ActionType, Decision
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import (
    AlwaysMarkBadDetector,
    AlwaysNoneDetector,
    FixedDecisionDetector,
    get_sample_value,
    make_test_controller,
)


async def _raise_runtime_error(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("notifier broken")


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
        assert harness.controller._active_run_id is None


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

    @pytest.mark.asyncio
    async def test_new_run_cleans_multiple_old_scrape_targets(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=4,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        await harness.controller.register_rank(
            run_id="run-1", rank=1, world_size=4,
            node_id="node-1", exporter_address="http://node-1:9090",
        )
        assert "rank-0" in harness.metric_store._scrape_targets
        assert "rank-1" in harness.metric_store._scrape_targets

        await harness.controller.register_rank(
            run_id="run-2", rank=2, world_size=4,
            node_id="node-2", exporter_address="http://node-2:9090",
        )

        assert "rank-0" not in harness.metric_store._scrape_targets
        assert "rank-1" not in harness.metric_store._scrape_targets
        assert "rank-2" in harness.metric_store._scrape_targets

    @pytest.mark.asyncio
    async def test_partial_registration_tick_still_runs(self, caplog: pytest.LogCaptureFixture) -> None:
        """world_size=4 but only 3 ranks register; tick runs normally
        but emits a WARNING about incomplete registration.
        """
        harness = make_test_controller()

        for rank in range(3):
            await harness.controller.register_rank(
                run_id="run-1", rank=rank, world_size=4,
                node_id=f"node-{rank}", exporter_address=f"http://node-{rank}:9090",
            )

        assert len(harness.controller._rank_placement) == 3
        assert 3 not in harness.controller._rank_placement
        assert harness.controller._expected_world_size == 4

        with caplog.at_level(logging.WARNING):
            await harness.controller._tick()

        assert harness.controller._tick_count == 1
        assert "incomplete_rank_registration" in caplog.text
        assert "registered=3" in caplog.text
        assert "expected=4" in caplog.text


    @pytest.mark.asyncio
    async def test_full_registration_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """All 4/4 ranks registered — tick should not emit WARNING."""
        harness = make_test_controller()

        for rank in range(4):
            await harness.controller.register_rank(
                run_id="run-1", rank=rank, world_size=4,
                node_id=f"node-{rank}", exporter_address=f"http://node-{rank}:9090",
            )

        assert harness.controller._expected_world_size == 4
        assert len(harness.controller._rank_placement) == 4

        with caplog.at_level(logging.WARNING):
            await harness.controller._tick()

        assert "incomplete_rank_registration" not in caplog.text

    @pytest.mark.asyncio
    async def test_expected_world_size_reset_on_new_run(self) -> None:
        """When a new run_id arrives, _expected_world_size is reset."""
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=8,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        assert harness.controller._expected_world_size == 8

        await harness.controller.register_rank(
            run_id="run-2", rank=0, world_size=4,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        assert harness.controller._expected_world_size == 4
        assert harness.controller._rank_placement == {0: "node-0"}


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


class TestTrainingJobStatusExporter:
    """Verify training job status is pushed to the ControllerExporter gauge."""

    @pytest.mark.asyncio
    async def test_tick_updates_training_job_status_gauge(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        harness = make_test_controller(controller_exporter=exporter)

        await harness.controller._tick()

        assert get_sample_value(registry, "ft_training_job_status") == 1.0

    @pytest.mark.asyncio
    async def test_failed_status_maps_to_negative(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        harness = make_test_controller(
            status_sequence=[JobStatus.FAILED],
            controller_exporter=exporter,
        )

        await harness.controller._tick()

        assert get_sample_value(registry, "ft_training_job_status") == -1.0

    @pytest.mark.asyncio
    async def test_stopped_status_maps_to_zero(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        harness = make_test_controller(
            status_sequence=[JobStatus.STOPPED],
            controller_exporter=exporter,
        )

        await harness.controller._tick()

        assert get_sample_value(registry, "ft_training_job_status") == 0.0

    @pytest.mark.asyncio
    async def test_pending_status_maps_to_two(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        harness = make_test_controller(
            status_sequence=[JobStatus.PENDING],
            controller_exporter=exporter,
        )

        await harness.controller._tick()

        assert get_sample_value(registry, "ft_training_job_status") == 2.0

    @pytest.mark.asyncio
    async def test_tick_count_incremented(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        harness = make_test_controller(controller_exporter=exporter)

        await harness.controller._tick()
        await harness.controller._tick()

        assert get_sample_value(registry, "ft_controller_tick_count_total") == 2.0


class TestExecuteDecision:
    @pytest.mark.asyncio
    async def test_none_decision_is_noop(self) -> None:
        harness = make_test_controller()
        await harness.controller._tick()

    @pytest.mark.asyncio
    async def test_mark_bad_and_restart_does_not_raise(self) -> None:
        harness = make_test_controller(
            detectors=[AlwaysMarkBadDetector()],
        )
        await harness.controller._tick()
        assert harness.controller._tick_count == 1

    @pytest.mark.asyncio
    async def test_enter_recovery_does_not_raise(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test recovery",
        ))
        harness = make_test_controller(detectors=[detector])
        await harness.controller._tick()
        assert harness.controller._tick_count == 1

    @pytest.mark.asyncio
    async def test_notify_human_sends_notification(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="test notify",
        ))
        harness = make_test_controller(detectors=[detector])
        await harness.controller._tick()

        assert harness.controller._tick_count == 1
        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 1
        title, content, severity = harness.notifier.calls[0]
        assert title == "Fault Alert"
        assert content == "test notify"
        assert severity == "critical"

    @pytest.mark.asyncio
    async def test_notify_human_without_notifier(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="test notify no notifier",
        ))
        harness = make_test_controller(detectors=[detector], notifier=None)
        await harness.controller._tick()
        assert harness.controller._tick_count == 1

    @pytest.mark.asyncio
    async def test_none_decision_does_not_notify(self) -> None:
        harness = make_test_controller(detectors=[AlwaysNoneDetector()])
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 0

    @pytest.mark.asyncio
    async def test_mark_bad_does_not_notify(self) -> None:
        harness = make_test_controller(detectors=[AlwaysMarkBadDetector()])
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 0

    @pytest.mark.asyncio
    async def test_enter_recovery_does_not_notify(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test recovery",
        ))
        harness = make_test_controller(detectors=[detector])
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 0

    @pytest.mark.asyncio
    async def test_notify_human_notifier_exception_does_not_crash(self) -> None:
        harness = make_test_controller(
            detectors=[FixedDecisionDetector(decision=Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason="test with broken notifier",
            ))],
        )
        assert harness.notifier is not None
        harness.notifier.send = _raise_runtime_error
        await harness.controller._tick()
        assert harness.controller._tick_count == 1

    @pytest.mark.asyncio
    async def test_notify_human_sends_on_every_tick(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="persistent fault",
        ))
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 2
