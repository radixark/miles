from __future__ import annotations

import asyncio
import logging

import pytest
from prometheus_client import CollectorRegistry

import miles.utils.ft.metric_names as mn
from miles.utils.ft.controller.controller_exporter import ControllerExporter
from miles.utils.ft.models import ActionType, Decision, RecoveryPhase, TriggerType
from miles.utils.ft.platform.protocols import JobStatus
from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.diagnostics.scheduler import DiagnosticScheduler
from miles.utils.ft.controller.mini_wandb import MiniWandb
from tests.fast.utils.ft.conftest import (
    AlwaysMarkBadDetector,
    AlwaysNoneDetector,
    FakeNodeManager,
    FakeTrainingJob,
    FixedDecisionDetector,
    get_sample_value,
    make_detector_context,
    make_fake_metric_store,
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
        ctx = make_detector_context(metric_store=harness.metric_store, mini_wandb=harness.mini_wandb)
        decision = harness.controller._evaluate_detectors(ctx)
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

    @pytest.mark.asyncio
    async def test_register_rank_stores_pid(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
            pid=1234,
        )

        assert harness.controller._rank_pids == {0: 1234}

    @pytest.mark.asyncio
    async def test_new_run_id_clears_rank_pids(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
            pid=1234,
        )
        assert harness.controller._rank_pids == {0: 1234}

        await harness.controller.register_rank(
            run_id="run-2", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
            pid=5678,
        )
        assert harness.controller._rank_pids == {0: 5678}

    @pytest.mark.asyncio
    async def test_register_rank_without_pid_does_not_store(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        assert harness.controller._rank_pids == {}


class TestGetRankPidsForNode:
    @pytest.mark.asyncio
    async def test_returns_pids_for_matching_node(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=4,
            node_id="node-0", exporter_address="http://node-0:9090",
            pid=100,
        )
        await harness.controller.register_rank(
            run_id="run-1", rank=1, world_size=4,
            node_id="node-0", exporter_address="http://node-0:9091",
            pid=101,
        )
        await harness.controller.register_rank(
            run_id="run-1", rank=2, world_size=4,
            node_id="node-1", exporter_address="http://node-1:9090",
            pid=200,
        )

        result = harness.controller._get_rank_pids_for_node("node-0")
        assert result == {0: 100, 1: 101}

    @pytest.mark.asyncio
    async def test_returns_empty_for_unknown_node(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
            pid=100,
        )

        result = harness.controller._get_rank_pids_for_node("node-999")
        assert result == {}

    @pytest.mark.asyncio
    async def test_excludes_ranks_without_pid(self) -> None:
        harness = make_test_controller()

        await harness.controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
            pid=100,
        )
        await harness.controller.register_rank(
            run_id="run-1", rank=1, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9091",
        )

        result = harness.controller._get_rank_pids_for_node("node-0")
        assert result == {0: 100}


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

    @pytest.mark.asyncio
    async def test_run_starts_and_stops_scrape_loop(self) -> None:
        """run() must start the scrape loop as a background task and
        stop it on shutdown, even if the metric store supports start/stop."""
        harness = make_test_controller(tick_interval=0.01)
        store = harness.metric_store

        started = False
        stopped = False
        original_start = store.start
        original_stop = store.stop

        async def tracking_start() -> None:
            nonlocal started
            started = True
            await original_start()

        async def tracking_stop() -> None:
            nonlocal stopped
            stopped = True
            await original_stop()

        store.start = tracking_start
        store.stop = tracking_stop

        async def _shutdown_soon() -> None:
            await asyncio.sleep(0.05)
            await harness.controller.shutdown()

        shutdown_task = asyncio.create_task(_shutdown_soon())
        await harness.controller.run()
        await shutdown_task

        assert started
        assert stopped


class TestDetectorChain:
    def test_first_non_none_wins(self) -> None:
        none_detector = AlwaysNoneDetector()
        bad_detector = AlwaysMarkBadDetector()
        harness = make_test_controller(
            detectors=[none_detector, bad_detector],
        )

        ctx = make_detector_context(metric_store=harness.metric_store, mini_wandb=harness.mini_wandb)
        decision = harness.controller._evaluate_detectors(ctx)
        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert none_detector.call_count == 1
        assert bad_detector.call_count == 1

    def test_short_circuit_after_non_none(self) -> None:
        bad_detector = AlwaysMarkBadDetector()
        trailing_detector = AlwaysNoneDetector()
        harness = make_test_controller(
            detectors=[bad_detector, trailing_detector],
        )

        ctx = make_detector_context(metric_store=harness.metric_store, mini_wandb=harness.mini_wandb)
        decision = harness.controller._evaluate_detectors(ctx)
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

        assert get_sample_value(registry, mn.TRAINING_JOB_STATUS) == 1.0

    @pytest.mark.asyncio
    async def test_failed_status_maps_to_negative(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        harness = make_test_controller(
            status_sequence=[JobStatus.FAILED],
            controller_exporter=exporter,
        )

        await harness.controller._tick()

        assert get_sample_value(registry, mn.TRAINING_JOB_STATUS) == -1.0

    @pytest.mark.asyncio
    async def test_stopped_status_maps_to_zero(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        harness = make_test_controller(
            status_sequence=[JobStatus.STOPPED],
            controller_exporter=exporter,
        )

        await harness.controller._tick()

        assert get_sample_value(registry, mn.TRAINING_JOB_STATUS) == 0.0

    @pytest.mark.asyncio
    async def test_pending_status_maps_to_two(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        harness = make_test_controller(
            status_sequence=[JobStatus.PENDING],
            controller_exporter=exporter,
        )

        await harness.controller._tick()

        assert get_sample_value(registry, mn.TRAINING_JOB_STATUS) == 2.0

    @pytest.mark.asyncio
    async def test_tick_count_incremented(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        harness = make_test_controller(controller_exporter=exporter)

        await harness.controller._tick()
        await harness.controller._tick()

        assert get_sample_value(registry, mn.CONTROLLER_TICK_COUNT + "_total") == 2.0


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
            trigger=TriggerType.CRASH,
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
            trigger=TriggerType.CRASH,
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


class TestMarkBadAndRestartReal:
    @pytest.mark.asyncio
    async def test_marks_bad_nodes(self) -> None:
        harness = make_test_controller(detectors=[AlwaysMarkBadDetector()])
        await harness.controller._tick()

        assert harness.node_manager.is_node_bad("node-1")

    @pytest.mark.asyncio
    async def test_stops_and_submits_training(self) -> None:
        harness = make_test_controller(detectors=[AlwaysMarkBadDetector()])
        await harness.controller._tick()

        assert harness.training_job._stopped
        assert harness.training_job._submitted

    @pytest.mark.asyncio
    async def test_clears_mini_wandb_before_submit(self) -> None:
        harness = make_test_controller(detectors=[AlwaysMarkBadDetector()])
        harness.mini_wandb.log_step(
            run_id="test", rank=0, step=1, metrics={"loss": 1.0},
        )
        assert harness.mini_wandb.latest(metric_name="loss", rank=0) == 1.0

        await harness.controller._tick()

        assert harness.mini_wandb.latest(metric_name="loss", rank=0) is None


class TestEnterRecovery:
    @pytest.mark.asyncio
    async def test_creates_recovery_orchestrator(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test recovery",
        ))
        harness = make_test_controller(detectors=[detector])
        assert harness.controller._recovery_orchestrator is None

        await harness.controller._tick()

        assert harness.controller._recovery_orchestrator is not None

    @pytest.mark.asyncio
    async def test_recovery_mode_skips_detectors(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test recovery",
        ))
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()
        initial_count = detector.call_count

        await harness.controller._tick()
        assert detector.call_count == initial_count

    @pytest.mark.asyncio
    async def test_recovery_complete_returns_to_monitoring(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test recovery",
        ))
        harness = make_test_controller(
            detectors=[detector],
            status_sequence=[JobStatus.RUNNING],
        )

        await harness.controller._tick()
        assert harness.controller._recovery_orchestrator is not None

        harness.controller._recovery_orchestrator._context.phase = RecoveryPhase.DONE

        await harness.controller._tick()
        assert harness.controller._recovery_orchestrator is None

    @pytest.mark.asyncio
    async def test_exporter_mode_reflects_recovery(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test recovery",
        ))
        harness = make_test_controller(
            detectors=[detector],
            controller_exporter=exporter,
        )

        await harness.controller._tick()

        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 0.0

        await harness.controller._tick()

        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 1.0


class TestGetStatus:
    def test_monitoring_mode_default(self) -> None:
        harness = make_test_controller()
        status = harness.controller.get_status()
        assert status["mode"] == "monitoring"
        assert status["recovery_phase"] is None
        assert status["tick_count"] == 0
        assert status["active_run_id"] is None
        assert status["bad_nodes"] == []

    @pytest.mark.asyncio
    async def test_monitoring_mode_after_tick(self) -> None:
        harness = make_test_controller()
        await harness.controller._tick()
        status = harness.controller.get_status()
        assert status["mode"] == "monitoring"
        assert status["tick_count"] == 1

    @pytest.mark.asyncio
    async def test_recovery_mode(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test",
        ))
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()
        await harness.controller._tick()
        status = harness.controller.get_status()

        assert status["mode"] == "recovery"
        assert status["recovery_phase"] is not None

    @pytest.mark.asyncio
    async def test_active_run_id_after_register(self) -> None:
        harness = make_test_controller()
        await harness.controller.register_rank(
            run_id="run-42", rank=0, world_size=1,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        status = harness.controller.get_status()
        assert status["active_run_id"] == "run-42"


class TestAgentManagement:
    def test_register_agent_adds_to_dict(self) -> None:
        harness = make_test_controller()
        agent = object()
        harness.controller.register_agent("node-0", agent)

        assert "node-0" in harness.controller._agents
        assert harness.controller._agents["node-0"] is agent

    def test_unregister_agent_removes_from_dict(self) -> None:
        harness = make_test_controller()
        agent = object()
        harness.controller.register_agent("node-0", agent)
        harness.controller.unregister_agent("node-0")

        assert "node-0" not in harness.controller._agents

    def test_unregister_nonexistent_agent_is_noop(self) -> None:
        harness = make_test_controller()
        harness.controller.unregister_agent("nonexistent")

    def test_register_overwrites_existing(self) -> None:
        harness = make_test_controller()
        agent1 = object()
        agent2 = object()
        harness.controller.register_agent("node-0", agent1)
        harness.controller.register_agent("node-0", agent2)

        assert harness.controller._agents["node-0"] is agent2


class TestDefaultDiagnosticSchedulerWiring:
    def test_default_scheduler_has_rank_pids_provider(self) -> None:
        controller = FtController(
            node_manager=FakeNodeManager(),
            training_job=FakeTrainingJob(),
            metric_store=make_fake_metric_store(),
            mini_wandb=MiniWandb(),
        )

        scheduler = controller._diagnostic_scheduler
        assert isinstance(scheduler, DiagnosticScheduler)
        assert scheduler._rank_pids_provider.__func__ is FtController._get_rank_pids_for_node
        assert scheduler._rank_pids_provider.__self__ is controller


class TestDefaultDiagnosticPipeline:
    def test_default_scheduler_has_gpu_pipeline(self) -> None:
        harness = make_test_controller()
        scheduler = harness.controller._diagnostic_scheduler
        assert "gpu" in scheduler._pipeline
