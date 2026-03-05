from __future__ import annotations

import asyncio
import logging

import pytest

import miles.utils.ft.metric_names as mn
from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.diagnostics.scheduler import DiagnosticScheduler
from miles.utils.ft.controller.metrics import start_metric_store_task, stop_metric_store_task
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.rank_registry import RankRegistry
from miles.utils.ft.models import ActionType, ControllerMode, Decision, RecoveryPhase
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import (
    FakeNodeManager,
    FakeTrainingJob,
    FixedDecisionDetector,
    get_sample_value,
    make_fake_metric_store,
    make_test_controller,
    make_test_exporter,
)


class TestTrainingJobStatusExporter:
    """Verify training job status is pushed to the ControllerExporter gauge."""

    @pytest.mark.asyncio
    async def test_tick_updates_training_job_status_gauge(self) -> None:
        registry, exporter = make_test_exporter()
        harness = make_test_controller(controller_exporter=exporter)

        await harness.controller._tick()

        assert get_sample_value(registry, mn.TRAINING_JOB_STATUS) == 1.0

    @pytest.mark.asyncio
    async def test_failed_status_maps_to_negative(self) -> None:
        registry, exporter = make_test_exporter()
        harness = make_test_controller(
            status_sequence=[JobStatus.FAILED],
            controller_exporter=exporter,
        )

        await harness.controller._tick()

        assert get_sample_value(registry, mn.TRAINING_JOB_STATUS) == -1.0

    @pytest.mark.asyncio
    async def test_stopped_status_maps_to_zero(self) -> None:
        registry, exporter = make_test_exporter()
        harness = make_test_controller(
            status_sequence=[JobStatus.STOPPED],
            controller_exporter=exporter,
        )

        await harness.controller._tick()

        assert get_sample_value(registry, mn.TRAINING_JOB_STATUS) == 0.0

    @pytest.mark.asyncio
    async def test_pending_status_maps_to_two(self) -> None:
        registry, exporter = make_test_exporter()
        harness = make_test_controller(
            status_sequence=[JobStatus.PENDING],
            controller_exporter=exporter,
        )

        await harness.controller._tick()

        assert get_sample_value(registry, mn.TRAINING_JOB_STATUS) == 2.0

    @pytest.mark.asyncio
    async def test_tick_count_incremented(self) -> None:
        registry, exporter = make_test_exporter()
        harness = make_test_controller(controller_exporter=exporter)

        await harness.controller._tick()
        await harness.controller._tick()

        assert get_sample_value(registry, mn.CONTROLLER_TICK_COUNT + "_total") == 2.0


class TestGetStatus:
    def test_monitoring_mode_default(self) -> None:
        harness = make_test_controller()
        status = harness.controller.get_status()
        assert status.mode == ControllerMode.MONITORING
        assert status.recovery_phase is None
        assert status.phase_history is None
        assert status.tick_count == 0
        assert status.active_run_id is None
        assert status.bad_nodes == []

    @pytest.mark.asyncio
    async def test_monitoring_mode_after_tick(self) -> None:
        harness = make_test_controller()
        await harness.controller._tick()
        status = harness.controller.get_status()
        assert status.mode == ControllerMode.MONITORING
        assert status.tick_count == 1

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

        assert status.mode == ControllerMode.RECOVERY
        assert status.recovery_phase is not None

    @pytest.mark.asyncio
    async def test_active_run_id_after_register(self) -> None:
        harness = make_test_controller()
        await harness.controller.register_rank(
            run_id="run-42", rank=0, world_size=1,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        status = harness.controller.get_status()
        assert status.active_run_id == "run-42"

    def test_recovery_in_progress_false_during_monitoring(self) -> None:
        harness = make_test_controller()
        status = harness.controller.get_status()
        assert status.recovery_in_progress is False
        assert status.bad_nodes_confirmed is False

    @pytest.mark.asyncio
    async def test_recovery_in_progress_true_during_recovery(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test",
        ))
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()
        await harness.controller._tick()
        status = harness.controller.get_status()

        assert status.recovery_in_progress is True

    @pytest.mark.asyncio
    async def test_bad_nodes_confirmed_false_in_early_phases(self) -> None:
        """During REATTEMPTING/MONITORING/DIAGNOSING, bad_nodes_confirmed is False."""
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test",
        ))
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()
        await harness.controller._tick()
        status = harness.controller.get_status()

        assert status.recovery_in_progress is True
        assert status.recovery_phase in {
            RecoveryPhase.CHECK_ALERTS,
            RecoveryPhase.REATTEMPTING,
            RecoveryPhase.MONITORING,
            RecoveryPhase.DIAGNOSING,
        }
        assert status.bad_nodes_confirmed is False

    def test_bad_nodes_confirmed_true_in_evict_phase(self) -> None:
        """When recovery phase is EVICT_AND_RESTART, bad_nodes_confirmed is True."""
        harness = make_test_controller()
        controller = harness.controller

        mock_orch = _MockOrchestrator(
            phase=RecoveryPhase.EVICT_AND_RESTART,
            phase_history=[RecoveryPhase.CHECK_ALERTS, RecoveryPhase.EVICT_AND_RESTART],
            bad_node_ids=["node-1"],
        )
        controller._recovery_orchestrator = mock_orch

        status = controller.get_status()
        assert status.recovery_in_progress is True
        assert status.bad_nodes_confirmed is True

    def test_bad_nodes_confirmed_true_in_notify_phase(self) -> None:
        harness = make_test_controller()
        controller = harness.controller

        mock_orch = _MockOrchestrator(
            phase=RecoveryPhase.NOTIFY,
            phase_history=[RecoveryPhase.CHECK_ALERTS, RecoveryPhase.NOTIFY],
            bad_node_ids=["node-2"],
        )
        controller._recovery_orchestrator = mock_orch

        status = controller.get_status()
        assert status.recovery_in_progress is True
        assert status.bad_nodes_confirmed is True

    def test_bad_nodes_confirmed_false_in_diagnosing_phase(self) -> None:
        harness = make_test_controller()
        controller = harness.controller

        mock_orch = _MockOrchestrator(
            phase=RecoveryPhase.DIAGNOSING,
            phase_history=[RecoveryPhase.CHECK_ALERTS, RecoveryPhase.DIAGNOSING],
            bad_node_ids=["node-3"],
        )
        controller._recovery_orchestrator = mock_orch

        status = controller.get_status()
        assert status.recovery_in_progress is True
        assert status.bad_nodes_confirmed is False


class _MockOrchestrator:
    """Minimal stand-in for RecoveryOrchestrator used in get_status() tests."""

    def __init__(
        self,
        *,
        phase: RecoveryPhase,
        phase_history: list[RecoveryPhase],
        bad_node_ids: list[str],
    ) -> None:
        self.phase = phase
        self.phase_history = phase_history
        self.bad_node_ids = bad_node_ids


class TestAgentManagement:
    def test_register_agent_adds_to_dict(self) -> None:
        harness = make_test_controller()
        agent = object()
        harness.controller.register_agent("node-0", agent)

        assert "node-0" in harness.controller._rank_registry.agents
        assert harness.controller._rank_registry.agents["node-0"] is agent

    def test_register_overwrites_existing(self) -> None:
        harness = make_test_controller()
        agent1 = object()
        agent2 = object()
        harness.controller.register_agent("node-0", agent1)
        harness.controller.register_agent("node-0", agent2)

        assert harness.controller._rank_registry.agents["node-0"] is agent2


class TestDefaultDiagnosticSchedulerWiring:
    def test_default_scheduler_has_rank_pids_provider(self) -> None:
        rank_registry = RankRegistry(
            mini_wandb=MiniWandb(),
        )
        controller = FtController(
            node_manager=FakeNodeManager(),
            training_job=FakeTrainingJob(),
            metric_store=make_fake_metric_store(),
            rank_registry=rank_registry,
        )

        scheduler = controller._diagnostic_scheduler
        assert isinstance(scheduler, DiagnosticScheduler)
        assert scheduler._rank_pids_provider.__func__ is RankRegistry.get_rank_pids_for_node
        assert scheduler._rank_pids_provider.__self__ is rank_registry


class TestDefaultDiagnosticPipeline:
    def test_default_scheduler_has_gpu_pipeline(self) -> None:
        harness = make_test_controller()
        scheduler = harness.controller._diagnostic_scheduler
        assert "gpu" in scheduler._pipeline


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


class TestScrapeLoopDefensiveBranches:
    @pytest.mark.asyncio
    async def test_scrape_loop_logs_error_on_crash(self, caplog: pytest.LogCaptureFixture) -> None:
        harness = make_test_controller()

        async def _crashing_start() -> None:
            raise RuntimeError("scrape exploded")

        harness.controller._metric_store.start = _crashing_start

        with caplog.at_level(logging.ERROR):
            task = await start_metric_store_task(harness.controller._metric_store)
            assert task is not None
            await asyncio.sleep(0.05)

        assert "scrape_loop_crashed" in caplog.text
