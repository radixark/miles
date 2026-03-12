"""Integration tests for training + rollout subsystems in FtController."""

from __future__ import annotations

from typing import NamedTuple

import pytest
from prometheus_client import CollectorRegistry

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.factories.controller import assemble_ft_controller
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.main.models import NormalSt
from miles.utils.ft.controller.state_machines.subsystem import DetectingAnomalySt, RecoveringSt
from miles.utils.ft.controller.state_machines.restart.models import MonitoringIterationProgressConfig, MonitoringSustainedAliveConfig
from miles.utils.ft.controller.subsystem_hub import RestartMode, SubsystemHub
from miles.utils.ft.controller.types import ActionType, Decision, MetricStore, TriggerType

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
    """Fires a configured decision once, then returns NONE forever.

    Needed because test fakes complete all async operations instantly (evict,
    stop, restart, monitor), so run_stepper_to_convergence can drive a
    subsystem through the full recovery cycle back to DetectingAnomalySt within
    a single tick.  A real detector would then re-fire on stale metrics that
    haven't been cleared yet, consuming cooldown budget repeatedly.

    In production this doesn't happen: recovery blocks on real I/O (job stays
    PENDING for seconds/minutes), so the convergence loop exits before the
    subsystem returns to DetectingAnomalySt within the same tick.
    """

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
    time_series_store: MiniPrometheus
    subsystem_hub: SubsystemHub


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
    time_series_store = MiniPrometheus(config=MiniPrometheusConfig())
    mini_wandb = MiniWandb()
    metric_store = MetricStore(time_series_store=time_series_store, mini_wandb=mini_wandb)
    notifier = FakeNotifier()
    controller_exporter = ControllerExporter(registry=CollectorRegistry())
    bundle = assemble_ft_controller(
        node_manager=node_manager,
        main_job=main_job,
        metric_store=metric_store,
        rollout_cell_ids=resolved_cell_ids,
        scrape_target_manager=time_series_store,
        notifier=notifier,
        detectors=training_detectors,
        tick_interval=0.01,
        controller_exporter=controller_exporter,
        diagnostic_orchestrator=diagnostic_orchestrator,
        recovery_cooldown_window_minutes=30.0,
        recovery_cooldown_max_count=3,
        registration_grace_ticks=0,
        monitoring_success_iterations=monitoring_success_iterations,
    )
    controller = bundle.controller
    subsystem_hub = bundle.subsystem_hub

    controller._activate_run("test-run")
    subsystem_hub.training_rank_roster.rank_placement[0] = "train-node-0"
    subsystem_hub.training_rank_roster.rank_placement[1] = "train-node-1"

    rollout_manager_handle = FakeRmHandle()
    subsystem_hub.set_rollout_handle(rollout_manager_handle)

    for cid in resolved_cell_ids:
        subsystem_hub.set_rollout_node_ids(cid, {f"rollout-node-{cid}-0", f"rollout-node-{cid}-1"})

    return _RolloutTestHarness(
        controller=controller,
        main_job=main_job,
        node_manager=node_manager,
        notifier=notifier,
        rollout_manager_handle=rollout_manager_handle,
        time_series_store=time_series_store,
        subsystem_hub=subsystem_hub,
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

    def test_rollout_spec_has_subsystem_restart_mode(self) -> None:
        controller, *_ = _make_test_controller_with_rollout()
        rollout_spec = controller._subsystem_specs["rollout_ep72"]
        assert rollout_spec.config.restart_mode == RestartMode.SUBSYSTEM

    def test_training_spec_has_main_job_restart_mode(self) -> None:
        controller, *_ = _make_test_controller_with_rollout()
        training_spec = controller._subsystem_specs["training"]
        assert training_spec.config.restart_mode == RestartMode.MAIN_JOB

    def test_rollout_active_node_ids_returns_registered_nodes(self) -> None:
        harness = _make_test_controller_with_rollout()
        rollout_spec = harness.controller._subsystem_specs["rollout_ep72"]
        assert rollout_spec.runtime.get_active_node_ids() == {"rollout-node-ep72-0", "rollout-node-ep72-1"}


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

        rollout_spec = controller._subsystem_specs["rollout_ep72"]
        crash_detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["rollout-node-ep72-0"],
                reason="rollout cell ep72 dead",
                trigger=TriggerType.CRASH,
            )
        )
        rollout_spec.config.detectors.clear()
        rollout_spec.config.detectors.append(crash_detector)

        await controller._tick()

        state = controller._state_machine.state
        assert isinstance(state, NormalSt)
        rollout_state = state.subsystems["rollout_ep72"]
        assert isinstance(rollout_state, RecoveringSt), (
            f"Expected Recovering, got {type(rollout_state).__name__}"
        )


class TestSubsystemSpecsIncludeRollout:
    def test_subsystem_specs_include_rollout(self) -> None:
        """After construction, subsystem_specs contains both training + rollout."""
        controller, *_ = _make_test_controller_with_rollout(
            cell_ids=["ep72", "ep36"],
        )

        specs = controller._subsystem_specs
        assert "training" in specs
        assert "rollout_ep72" in specs
        assert "rollout_ep36" in specs
        assert len(specs) == 3

    def test_subsystem_specs_have_correct_actuator_and_config_types(self) -> None:
        from miles.utils.ft.adapters.impl.ray.rollout_actuator import RayRolloutActuator
        from miles.utils.ft.adapters.impl.training_actuator import TrainingSubsystemActuator

        controller, *_ = _make_test_controller_with_rollout()
        specs = controller._subsystem_specs

        training = specs["training"]
        assert isinstance(training.runtime.actuator, TrainingSubsystemActuator)
        assert training.config.restart_mode == RestartMode.MAIN_JOB
        assert isinstance(training.config.monitoring_config, MonitoringIterationProgressConfig)

        rollout = specs["rollout_ep72"]
        assert isinstance(rollout.runtime.actuator, RayRolloutActuator)
        assert rollout.config.restart_mode == RestartMode.SUBSYSTEM
        assert isinstance(rollout.config.monitoring_config, MonitoringSustainedAliveConfig)
        assert len(rollout.config.detectors) > 0


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

        rollout_spec = controller._subsystem_specs["rollout_ep72"]
        rollout_spec.config.monitoring_config = MonitoringSustainedAliveConfig(
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
        rollout_spec.config.detectors.clear()
        rollout_spec.config.detectors.append(rollout_detector)

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

        rollout_spec = controller._subsystem_specs["rollout_ep72"]
        rollout_spec.config.monitoring_config = MonitoringSustainedAliveConfig(
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
        rollout_spec.config.detectors.clear()
        rollout_spec.config.detectors.append(rollout_detector)

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

        harness.subsystem_hub.training_rank_roster.rank_placement[0] = shared_node
        harness.subsystem_hub.training_rank_roster.rank_placement[1] = "train-node-1"

        rollout_spec = controller._subsystem_specs["rollout_ep72"]
        rollout_spec.config.monitoring_config = MonitoringSustainedAliveConfig(
            alive_duration_seconds=0,
            timeout_seconds=60,
        )

        harness.subsystem_hub.set_rollout_node_ids("ep72", {shared_node, "rollout-node-ep72-0"})

        rollout_detector = _OneShotDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=[shared_node],
                reason="rollout fault on shared node",
                trigger=TriggerType.CRASH,
            )
        )
        rollout_spec.config.detectors.clear()
        rollout_spec.config.detectors.append(rollout_detector)

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
