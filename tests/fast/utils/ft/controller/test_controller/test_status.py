from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pytest

import miles.utils.ft.models.metric_names as mn
from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator
from miles.utils.ft.controller.main_state_machine import DetectingAnomaly, Recovering
from miles.utils.ft.controller.metrics.lifecycle import start_metric_store_task, stop_metric_store_task
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery.recovery_stepper import (
    EvictingAndRestarting,
    NotifyHumans,
    RealtimeChecks,
    RecoveryDone,
    StopTimeDiagnostics,
)
from miles.utils.ft.controller.recovery.restart_stepper import Evicting
from miles.utils.ft.models.recovery import ControllerMode
from miles.utils.ft.protocols.platform import JobStatus
from tests.fast.utils.ft.conftest import (
    AlwaysEnterRecoveryDetector,
    FakeNodeManager,
    FakeTrainingJob,
    get_sample_value,
    make_fake_metric_store,
    make_test_controller,
    make_test_exporter,
    run_controller_briefly,
)


class TestTrainingJobStatusExporter:
    @pytest.mark.parametrize("status_sequence,expected_value", [
        ([JobStatus.RUNNING], 1.0),
        ([JobStatus.FAILED], -1.0),
        ([JobStatus.STOPPED], 0.0),
        ([JobStatus.PENDING], 2.0),
    ])
    @pytest.mark.asyncio
    async def test_status_maps_to_gauge_value(
        self, status_sequence: list[JobStatus], expected_value: float,
    ) -> None:
        registry, exporter = make_test_exporter()
        harness = make_test_controller(
            status_sequence=status_sequence,
            controller_exporter=exporter,
        )

        await harness.controller._tick()

        assert get_sample_value(registry, mn.TRAINING_JOB_STATUS) == expected_value

    @pytest.mark.asyncio
    async def test_tick_count_incremented(self) -> None:
        registry, exporter = make_test_exporter()
        harness = make_test_controller(controller_exporter=exporter)

        await harness.controller._tick()
        await harness.controller._tick()

        assert get_sample_value(registry, mn.CONTROLLER_TICK_COUNT + "_total") == 2.0


class TestGetStatus:
    def test_monitoring_mode_default(self) -> None:
        harness = make_test_controller(register_dummy_rank=False)
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
        detector = AlwaysEnterRecoveryDetector(reason="test")
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()
        await harness.controller._tick()
        status = harness.controller.get_status()

        assert status.mode == ControllerMode.RECOVERY
        assert status.recovery_phase is not None

    @pytest.mark.asyncio
    async def test_active_run_id_after_register(self) -> None:
        harness = make_test_controller()
        harness.controller._activate_run("run-42")
        harness.controller.rank_roster.register_training_rank(
            run_id="run-42", rank=0, world_size=1,
            node_id="node-0", exporter_address="http://node-0:9090",
            pid=1,
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
        detector = AlwaysEnterRecoveryDetector(reason="test")
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()
        await harness.controller._tick()
        status = harness.controller.get_status()

        assert status.recovery_in_progress is True

    @pytest.mark.asyncio
    async def test_bad_nodes_confirmed_false_in_early_phases(self) -> None:
        """During RealtimeChecks / DirectlyRestarting / StopTimeDiagnostics, bad_nodes_confirmed is False."""
        detector = AlwaysEnterRecoveryDetector(reason="test")
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()
        await harness.controller._tick()
        status = harness.controller.get_status()

        assert status.recovery_in_progress is True
        assert status.bad_nodes_confirmed is False

    def test_bad_nodes_confirmed_when_evicting(self) -> None:
        harness = make_test_controller()
        harness.controller._state_machine._state = Recovering(
            recovery=EvictingAndRestarting(
                restart=Evicting(bad_node_ids=["node-0"]),
            ),
            trigger="crash",
            recovery_start_time=datetime.now(timezone.utc),
        )
        status = harness.controller.get_status()
        assert status.recovery_in_progress is True
        assert status.bad_nodes_confirmed is True
        assert status.bad_nodes == ["node-0"]

    def test_bad_nodes_confirmed_when_notifying(self) -> None:
        harness = make_test_controller()
        harness.controller._state_machine._state = Recovering(
            recovery=NotifyHumans(state_before="Test"),
            trigger="crash",
            recovery_start_time=datetime.now(timezone.utc),
        )
        status = harness.controller.get_status()
        assert status.bad_nodes_confirmed is True

    def test_bad_nodes_not_confirmed_during_diagnostics(self) -> None:
        harness = make_test_controller()
        harness.controller._state_machine._state = Recovering(
            recovery=StopTimeDiagnostics(),
            trigger="crash",
            recovery_start_time=datetime.now(timezone.utc),
        )
        status = harness.controller.get_status()
        assert status.bad_nodes_confirmed is False


class TestAgentManagement:
    def test_register_agent_adds_to_dict(self) -> None:
        harness = make_test_controller()
        agent = object()
        harness.controller.register_node_agent("node-0", agent)

        assert "node-0" in harness.controller._agents
        assert harness.controller._agents["node-0"] is agent

    def test_register_overwrites_existing(self) -> None:
        harness = make_test_controller()
        agent1 = object()
        agent2 = object()
        harness.controller.register_node_agent("node-0", agent1)
        harness.controller.register_node_agent("node-0", agent2)

        assert harness.controller._agents["node-0"] is agent2


class TestDefaultDiagnosticOrchestratorWiring:
    def test_default_orchestrator_has_rank_pids_provider(self) -> None:
        controller = FtController.create(
            node_manager=FakeNodeManager(),
            training_job=FakeTrainingJob(),
            metric_store=make_fake_metric_store(),
            mini_wandb=MiniWandb(),
        )

        assert callable(controller._platform_deps.rank_pids_provider)


class TestDefaultDiagnosticPipeline:
    def test_default_orchestrator_has_gpu_pipeline(self) -> None:
        harness = make_test_controller()
        orchestrator = harness.controller._platform_deps.diagnostic_orchestrator
        assert "gpu" in orchestrator._pipeline


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_stops_run_loop(self) -> None:
        harness = make_test_controller(tick_interval=0.01)

        await run_controller_briefly(harness, delay=0.05)

        assert harness.controller._shutting_down
        assert harness.controller._tick_count >= 1

    @pytest.mark.asyncio
    async def test_run_starts_and_stops_scrape_loop(self) -> None:
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

        await run_controller_briefly(harness, delay=0.05)

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
