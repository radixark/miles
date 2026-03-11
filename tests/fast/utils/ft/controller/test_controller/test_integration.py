"""Integration tests for training + rollout subsystems in FtController."""

from __future__ import annotations

from typing import NamedTuple

import pytest
from prometheus_client import CollectorRegistry

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.factory import create_ft_controller
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.main.models import NormalSt
from miles.utils.ft.controller.state_machines.subsystem import DetectingAnomalySt, RecoveringSt
from miles.utils.ft.controller.subsystem import MonitoringIterationProgressConfig, MonitoringSustainedAliveConfig, RestartMode
from miles.utils.ft.controller.subsystem_hub import SubsystemHub
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle
from tests.fast.utils.ft.conftest import (
    FakeDiagnosticOrchestrator,
    FakeMainJob,
    FakeNodeManager,
    FakeNotifier,
    FixedDecisionDetector,
)


# ---------------------------------------------------------------------------
# Rollout fakes
# ---------------------------------------------------------------------------


class FakeRmHandle:
    """Fake Ray remote handle for RolloutManager."""

    def __init__(self) -> None:
        self.stop_cell = _FakeRemoteMethod()
        self.start_cell = _FakeRemoteMethod(result=1)
        self.get_cell_status = _FakeRemoteMethod(result=JobStatus.RUNNING)


class _FakeRemoteMethod:
    def __init__(self, *, result: object = None) -> None:
        self._result = result
        self.call_count = 0
        self.call_args: list[tuple[object, ...]] = []

    async def remote(self, *args: object, **kwargs: object) -> object:
        self.call_count += 1
        self.call_args.append(args)
        return self._result


class _OneShotDecisionDetector(BaseFaultDetector):
    """Fires a configured decision once, then returns NONE forever."""

    def __init__(self, decision: Decision) -> None:
        self._decision = decision
        self._fired = False

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        if not self._fired:
            self._fired = True
            return self._decision
        return Decision(action=ActionType.NONE, reason="already fired")


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


class _RolloutTestHarness(NamedTuple):
    controller: FtController
    main_job: FakeMainJob
    node_manager: FakeNodeManager
    notifier: FakeNotifier
    rollout_manager_handle: FakeRmHandle
    metric_store: MiniPrometheus
    hub: SubsystemHub


def _make_test_controller_with_rollout(
    *,
    training_detectors: list[BaseFaultDetector] | None = None,
    cell_ids: list[str] | None = None,
    monitoring_success_iterations: int = 10,
    diagnostic_orchestrator: object | None = None,
) -> _RolloutTestHarness:
    resolved_cell_ids = cell_ids or ["ep72"]

    node_manager = FakeNodeManager()
    main_job = FakeMainJob()
    metric_store = MiniPrometheus(config=MiniPrometheusConfig())
    mini_wandb = MiniWandb()
    notifier = FakeNotifier()
    controller_exporter = ControllerExporter(registry=CollectorRegistry())
    cooldown = SlidingWindowThrottle(window_minutes=30.0, max_count=3)

    bundle = create_ft_controller(
        node_manager=node_manager,
        main_job=main_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        rollout_cell_ids=resolved_cell_ids,
        scrape_target_manager=metric_store,
        notifier=notifier,
        detectors=training_detectors,
        tick_interval=0.01,
        controller_exporter=controller_exporter,
        diagnostic_orchestrator=diagnostic_orchestrator,
        recovery_cooldown=cooldown,
        registration_grace_ticks=0,
        monitoring_success_iterations=monitoring_success_iterations,
    )
    controller = bundle.controller
    hub = bundle.hub

    controller._activate_run("test-run")
    controller.training_rank_roster.rank_placement[0] = "train-node-0"
    controller.training_rank_roster.rank_placement[1] = "train-node-1"

    rollout_manager_handle = FakeRmHandle()
    hub.set_rollout_handle(rollout_manager_handle)

    return _RolloutTestHarness(
        controller=controller,
        main_job=main_job,
        node_manager=node_manager,
        notifier=notifier,
        rollout_manager_handle=rollout_manager_handle,
        metric_store=metric_store,
        hub=hub,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegisterRolloutSubsystems:
    def test_rollout_entries_present_in_normal_state(self) -> None:
        controller, *_ = _make_test_controller_with_rollout(
            cell_ids=["ep72", "ep36"],
        )
        state = controller._state_machine.state
        assert isinstance(state, NormalSt)
        assert "training" in state.subsystems
        assert "rollout_ep72" in state.subsystems
        assert "rollout_ep36" in state.subsystems
        assert len(state.subsystems) == 3

    def test_rollout_config_has_subsystem_restart_mode(self) -> None:
        controller, *_ = _make_test_controller_with_rollout()
        rollout_config = controller._tick_loop.subsystem_configs["rollout_ep72"]
        assert rollout_config.restart_mode == RestartMode.SUBSYSTEM

    def test_training_config_has_main_job_restart_mode(self) -> None:
        controller, *_ = _make_test_controller_with_rollout()
        training_config = controller._tick_loop.subsystem_configs["training"]
        assert training_config.restart_mode == RestartMode.MAIN_JOB

    def test_rollout_active_node_ids_defaults_to_empty(self) -> None:
        controller, *_ = _make_test_controller_with_rollout()
        rollout_config = controller._tick_loop.subsystem_configs["rollout_ep72"]
        assert rollout_config.get_active_node_ids() == set()


class TestNormalOperationWithRollout:
    @pytest.mark.anyio
    async def test_all_subsystems_healthy_stays_in_normal_state(self) -> None:
        """Training + 2 rollout cells, no detectors fire -> NormalState stays."""
        controller, *_ = _make_test_controller_with_rollout(
            cell_ids=["ep72", "ep36"],
        )

        for _ in range(3):
            await controller._tick()

        state = controller._state_machine.state
        assert isinstance(state, NormalSt)
        assert "training" in state.subsystems
        assert "rollout_ep72" in state.subsystems
        assert "rollout_ep36" in state.subsystems

    @pytest.mark.anyio
    async def test_initial_subsystem_states_are_detecting_anomaly(self) -> None:
        controller, *_ = _make_test_controller_with_rollout()

        state = controller._state_machine.state
        assert isinstance(state, NormalSt)

        for name, sub_state in state.subsystems.items():
            assert isinstance(sub_state, DetectingAnomalySt), (
                f"Expected DetectingAnomaly for {name}, got {type(sub_state).__name__}"
            )


class TestRolloutCrashRecovery:
    @pytest.mark.anyio
    async def test_rollout_detector_fires_enters_recovery(self) -> None:
        """When a rollout detector fires, the rollout subsystem enters Recovering."""
        controller, *_ = _make_test_controller_with_rollout()

        rollout_config = controller._tick_loop.subsystem_configs["rollout_ep72"]
        crash_detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["rollout-node-ep72-0"],
                reason="rollout cell ep72 dead",
                trigger=TriggerType.CRASH,
            )
        )
        rollout_config.detectors.clear()
        rollout_config.detectors.append(crash_detector)

        await controller._tick()

        state = controller._state_machine.state
        assert isinstance(state, NormalSt)
        rollout_state = state.subsystems["rollout_ep72"]
        assert isinstance(rollout_state, RecoveringSt), (
            f"Expected Recovering, got {type(rollout_state).__name__}"
        )


class TestSubsystemConfigsIncludeRollout:
    def test_subsystem_configs_include_rollout(self) -> None:
        """After construction, subsystem_configs contains both training + rollout."""
        controller, *_ = _make_test_controller_with_rollout(
            cell_ids=["ep72", "ep36"],
        )

        configs = controller._tick_loop.subsystem_configs
        assert "training" in configs
        assert "rollout_ep72" in configs
        assert "rollout_ep36" in configs
        assert len(configs) == 3

    def test_subsystem_configs_have_correct_actuator_and_config_types(self) -> None:
        from miles.utils.ft.adapters.impl.ray.rollout_actuator import RayRolloutActuator
        from miles.utils.ft.adapters.impl.training_actuator import TrainingSubsystemActuator

        controller, *_ = _make_test_controller_with_rollout()
        configs = controller._tick_loop.subsystem_configs

        training = configs["training"]
        assert isinstance(training.actuator, TrainingSubsystemActuator)
        assert training.restart_mode == RestartMode.MAIN_JOB
        assert isinstance(training.monitoring_config, MonitoringIterationProgressConfig)

        rollout = configs["rollout_ep72"]
        assert isinstance(rollout.actuator, RayRolloutActuator)
        assert rollout.restart_mode == RestartMode.SUBSYSTEM
        assert isinstance(rollout.monitoring_config, MonitoringSustainedAliveConfig)
        assert len(rollout.detectors) > 0


class TestStatusReportsRollout:
    def test_status_includes_rollout_subsystem_states(self) -> None:
        controller, *_ = _make_test_controller_with_rollout(
            cell_ids=["ep72", "ep36"],
        )

        status = controller.get_status()
        assert "rollout_ep72" in status.subsystem_states
        assert "rollout_ep36" in status.subsystem_states
        assert status.subsystem_states["rollout_ep72"] == "DetectingAnomalySt"

    def test_status_training_only_has_no_rollout_keys(self) -> None:
        from tests.fast.utils.ft.conftest import make_test_controller

        harness = make_test_controller()
        status = harness.controller.get_status()
        rollout_keys = [k for k in status.subsystem_states if k.startswith("rollout_")]
        assert rollout_keys == []


class TestFullLevel1RecoveryCycle:
    @pytest.mark.anyio
    async def test_rollout_completes_full_recovery_and_returns_to_detecting_anomaly(self) -> None:
        """Detect -> RealtimeChecks -> Evict -> Stop+Start -> Monitor(sustained_alive) -> Done."""
        harness = _make_test_controller_with_rollout(
            diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        )
        controller = harness.controller

        rollout_config = controller._tick_loop.subsystem_configs["rollout_ep72"]
        rollout_config.monitoring_config = MonitoringSustainedAliveConfig(
            alive_duration_seconds=0,
            timeout_seconds=60,
        )

        rollout_detector = _OneShotDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["rollout-node-ep72-0"],
                reason="rollout cell ep72 dead",
                trigger=TriggerType.CRASH,
            )
        )
        rollout_config.detectors.clear()
        rollout_config.detectors.append(rollout_detector)

        await controller._tick()

        state = controller._state_machine.state
        assert isinstance(state, NormalSt)
        assert isinstance(state.subsystems["rollout_ep72"], DetectingAnomalySt)
        assert isinstance(state.subsystems["training"], DetectingAnomalySt)

        assert harness.node_manager.was_ever_marked_bad("rollout-node-ep72-0")
        assert harness.rollout_manager_handle.stop_cell.call_count == 1
        assert harness.rollout_manager_handle.start_cell.call_count == 1


class TestLevel1FailureEscalation:
    @pytest.mark.anyio
    async def test_level1_restart_failure_escalates_to_notify_humans(self) -> None:
        """L1 restart fails (FAILED status) -> StopTimeDiagnostics -> NotifyHumans -> notifier called."""
        harness = _make_test_controller_with_rollout(
            diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        )
        controller = harness.controller

        rollout_config = controller._tick_loop.subsystem_configs["rollout_ep72"]
        rollout_config.monitoring_config = MonitoringSustainedAliveConfig(
            alive_duration_seconds=0,
            timeout_seconds=60,
        )

        harness.rollout_manager_handle.get_cell_status = _FakeRemoteMethod(result=JobStatus.FAILED)

        rollout_detector = _OneShotDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["rollout-node-ep72-0"],
                reason="rollout cell ep72 dead",
                trigger=TriggerType.CRASH,
            )
        )
        rollout_config.detectors.clear()
        rollout_config.detectors.append(rollout_detector)

        await controller._tick()

        state = controller._state_machine.state
        assert isinstance(state, NormalSt)
        assert isinstance(state.subsystems["rollout_ep72"], DetectingAnomalySt)
        assert harness.node_manager.was_ever_marked_bad("rollout-node-ep72-0")

        recovery_alerts = [
            (title, content)
            for title, content, _ in harness.notifier.calls
            if title == "Recovery Alert"
        ]
        assert len(recovery_alerts) >= 1, (
            f"Expected 'Recovery Alert' notification, got: {harness.notifier.calls}"
        )


class TestColocatedHardwareFault:
    @pytest.mark.anyio
    async def test_shared_bad_node_triggers_training_main_job_restart(self) -> None:
        """Training (no L1) and rollout (L1) share a bad node; training escalates to main job restart."""
        shared_node = "node-X"
        training_detector = _OneShotDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=[shared_node],
                reason="training fault on shared node",
                trigger=TriggerType.CRASH,
            )
        )

        harness = _make_test_controller_with_rollout(
            training_detectors=[training_detector],
            monitoring_success_iterations=0,
            diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        )
        controller = harness.controller

        controller.training_rank_roster.rank_placement[0] = shared_node
        controller.training_rank_roster.rank_placement[1] = "train-node-1"

        rollout_config = controller._tick_loop.subsystem_configs["rollout_ep72"]
        rollout_config.monitoring_config = MonitoringSustainedAliveConfig(
            alive_duration_seconds=0,
            timeout_seconds=60,
        )

        rollout_detector = _OneShotDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=[shared_node],
                reason="rollout fault on shared node",
                trigger=TriggerType.CRASH,
            )
        )
        rollout_config.detectors.clear()
        rollout_config.detectors.append(rollout_detector)

        await controller._tick()

        assert harness.node_manager.was_ever_marked_bad(shared_node)
        assert harness.main_job._stopped is True
        assert harness.main_job._submitted is True

        final_state = controller._state_machine.state
        assert isinstance(final_state, NormalSt)
        assert "training" in final_state.subsystems
        assert "rollout_ep72" in final_state.subsystems

        assert harness.rollout_manager_handle.stop_cell.call_count >= 1
        assert harness.rollout_manager_handle.start_cell.call_count >= 1
