"""Integration tests for M12: training + rollout subsystems in FtController."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple

import pytest
from prometheus_client import CollectorRegistry

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.controller import FtController, _RolloutSubsystemConfig, _build_rollout_subsystem_config
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.factory import create_ft_controller
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.main.models import NormalState, RestartingMainJobState
from miles.utils.ft.controller.state_machines.subsystem import DetectingAnomaly, Recovering
from miles.utils.ft.controller.subsystem import MonitoringIterationProgressConfig, MonitoringSustainedAliveConfig, RestartMode
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle
from tests.fast.utils.ft.conftest import (
    FakeDiagnosticOrchestrator,
    FakeMainJob,
    FakeNodeManager,
    FakeNotifier,
    FixedDecisionDetector,
    make_test_controller,
)


# ---------------------------------------------------------------------------
# Rollout fakes
# ---------------------------------------------------------------------------


class FakeRolloutCellAgent:
    """Mimics RolloutCellAgent for testing register_rollout_subsystems."""

    def __init__(self, *, cell_id: str, node_ids: set[str]) -> None:
        self._cell_id = cell_id
        self._node_ids = node_ids

    @property
    def cell_id(self) -> str:
        return self._cell_id

    def get_node_ids(self) -> set[str]:
        return set(self._node_ids)


class FakeRolloutAgent:
    """Mimics FtRolloutAgent for testing register_rollout_subsystems."""

    def __init__(self, cells: dict[str, FakeRolloutCellAgent]) -> None:
        self._cells = cells

    def get_cell_ids(self) -> list[str]:
        return list(self._cells.keys())

    def get_cell_agent(self, cell_id: str) -> FakeRolloutCellAgent:
        return self._cells[cell_id]


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
    rm_handle: FakeRmHandle
    metric_store: MiniPrometheus


def _make_test_controller_with_rollout(
    *,
    training_detectors: list[BaseFaultDetector] | None = None,
    cell_ids: list[str] | None = None,
    cell_node_ids: dict[str, set[str]] | None = None,
    monitoring_success_iterations: int = 10,
    diagnostic_orchestrator: object | None = None,
) -> _RolloutTestHarness:
    resolved_cell_ids = cell_ids or ["ep72"]
    resolved_cell_nodes = cell_node_ids or {
        cid: {f"rollout-node-{cid}-0", f"rollout-node-{cid}-1"}
        for cid in resolved_cell_ids
    }

    node_manager = FakeNodeManager()
    main_job = FakeMainJob()
    metric_store = MiniPrometheus(config=MiniPrometheusConfig())
    mini_wandb = MiniWandb()
    notifier = FakeNotifier()
    controller_exporter = ControllerExporter(registry=CollectorRegistry())
    cooldown = SlidingWindowThrottle(window_minutes=30.0, max_count=3)

    controller = create_ft_controller(
        node_manager=node_manager,
        main_job=main_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        rollout_num_cells=len(resolved_cell_ids),
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

    controller._activate_run("test-run")
    controller.training_rank_roster.rank_placement[0] = "train-node-0"
    controller.training_rank_roster.rank_placement[1] = "train-node-1"

    rm_handle = FakeRmHandle()
    cells = {
        cid: FakeRolloutCellAgent(cell_id=cid, node_ids=resolved_cell_nodes[cid])
        for cid in resolved_cell_ids
    }
    rollout_agent = FakeRolloutAgent(cells=cells)

    controller.register_rollout_subsystems(
        rm_handle=rm_handle,
        ft_rollout_agent=rollout_agent,
    )

    return _RolloutTestHarness(
        controller=controller,
        main_job=main_job,
        node_manager=node_manager,
        notifier=notifier,
        rm_handle=rm_handle,
        metric_store=metric_store,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegisterRolloutSubsystems:
    def test_registration_adds_rollout_entries_to_normal_state(self) -> None:
        controller, *_ = _make_test_controller_with_rollout(
            cell_ids=["ep72", "ep36"],
        )
        state = controller._state_machine.state
        assert isinstance(state, NormalState)
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

    def test_rollout_configs_stored_on_controller(self) -> None:
        controller, *_ = _make_test_controller_with_rollout(cell_ids=["ep72", "ep36"])
        assert len(controller._rollout_configs) == 2
        cell_ids = [c.cell_id for c in controller._rollout_configs]
        assert "ep72" in cell_ids
        assert "ep36" in cell_ids

    def test_rollout_config_returns_configured_node_ids(self) -> None:
        controller, *_ = _make_test_controller_with_rollout(
            cell_node_ids={"ep72": {"node-r0", "node-r1", "node-r2"}},
        )
        rollout_config = controller._tick_loop.subsystem_configs["rollout_ep72"]
        assert rollout_config.get_active_node_ids() == {"node-r0", "node-r1", "node-r2"}

    def test_register_in_non_normal_state_raises_runtime_error(self) -> None:
        harness = make_test_controller(rollout_num_cells=1)
        controller = harness.controller

        controller._state_machine.force_state(
            RestartingMainJobState(
                requestor_name="training",
                start_time=datetime.now(timezone.utc),
                requestor_frozen_state=DetectingAnomaly(),
            )
        )

        rm_handle = FakeRmHandle()
        cells = {"ep72": FakeRolloutCellAgent(cell_id="ep72", node_ids={"n1"})}
        rollout_agent = FakeRolloutAgent(cells=cells)

        with pytest.raises(RuntimeError, match="Cannot register rollout subsystems"):
            controller.register_rollout_subsystems(
                rm_handle=rm_handle,
                ft_rollout_agent=rollout_agent,
            )


class TestNormalOperationWithRollout:
    @pytest.mark.anyio
    async def test_all_subsystems_healthy_stays_in_normal_state(self) -> None:
        """Training + 2 rollout cells, no detectors fire → NormalState stays."""
        controller, *_ = _make_test_controller_with_rollout(
            cell_ids=["ep72", "ep36"],
        )

        for _ in range(3):
            await controller._tick()

        state = controller._state_machine.state
        assert isinstance(state, NormalState)
        assert "training" in state.subsystems
        assert "rollout_ep72" in state.subsystems
        assert "rollout_ep36" in state.subsystems

    @pytest.mark.anyio
    async def test_initial_subsystem_states_are_detecting_anomaly(self) -> None:
        controller, *_ = _make_test_controller_with_rollout()

        state = controller._state_machine.state
        assert isinstance(state, NormalState)

        for name, sub_state in state.subsystems.items():
            assert isinstance(sub_state, DetectingAnomaly), (
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
        assert isinstance(state, NormalState)
        rollout_state = state.subsystems["rollout_ep72"]
        assert isinstance(rollout_state, Recovering), (
            f"Expected Recovering, got {type(rollout_state).__name__}"
        )


class TestSubsystemConfigsIncludeRollout:
    def test_subsystem_configs_include_rollout_after_registration(self) -> None:
        """After rollout registration, subsystem_configs contains both training + rollout."""
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
        assert status.rollout_subsystem_states is not None
        assert "rollout_ep72" in status.rollout_subsystem_states
        assert "rollout_ep36" in status.rollout_subsystem_states
        assert status.rollout_subsystem_states["rollout_ep72"] == "DetectingAnomaly"

    def test_status_no_rollout_when_training_only(self) -> None:
        harness = make_test_controller()
        status = harness.controller.get_status()
        assert status.rollout_subsystem_states is None


class TestBuildRolloutSubsystemConfig:
    def test_config_has_correct_monitoring_config(self) -> None:
        rm_handle = FakeRmHandle()
        config = _RolloutSubsystemConfig(
            cell_id="ep72",
            rm_handle=rm_handle,
            get_active_node_ids=lambda: {"n1", "n2"},
        )
        subsystem_config = _build_rollout_subsystem_config(config=config)

        assert subsystem_config.restart_mode == RestartMode.SUBSYSTEM
        assert subsystem_config.get_active_node_ids() == {"n1", "n2"}
        assert isinstance(subsystem_config.monitoring_config, MonitoringSustainedAliveConfig)
        assert subsystem_config.monitoring_config.alive_duration_seconds == 180


class TestFullLevel1RecoveryCycle:
    @pytest.mark.anyio
    async def test_rollout_completes_full_recovery_and_returns_to_detecting_anomaly(self) -> None:
        """Detect → RealtimeChecks → Evict → Stop+Start → Monitor(sustained_alive) → Done."""
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
        assert isinstance(state, NormalState)
        assert isinstance(state.subsystems["rollout_ep72"], DetectingAnomaly)
        assert isinstance(state.subsystems["training"], DetectingAnomaly)

        assert harness.node_manager.was_ever_marked_bad("rollout-node-ep72-0")
        assert harness.rm_handle.stop_cell.call_count == 1
        assert harness.rm_handle.start_cell.call_count == 1


class TestLevel1FailureEscalation:
    @pytest.mark.anyio
    async def test_level1_restart_failure_escalates_to_notify_humans(self) -> None:
        """L1 restart fails (FAILED status) → StopTimeDiagnostics → NotifyHumans → notifier called."""
        harness = _make_test_controller_with_rollout(
            diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        )
        controller = harness.controller

        rollout_config = controller._tick_loop.subsystem_configs["rollout_ep72"]
        rollout_config.monitoring_config = MonitoringSustainedAliveConfig(
            alive_duration_seconds=0,
            timeout_seconds=60,
        )

        harness.rm_handle.get_cell_status = _FakeRemoteMethod(result=JobStatus.FAILED)

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
        assert isinstance(state, NormalState)
        assert isinstance(state.subsystems["rollout_ep72"], DetectingAnomaly)
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
            cell_node_ids={"ep72": {shared_node, "rollout-node-0"}},
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
        assert isinstance(final_state, NormalState)
        assert "training" in final_state.subsystems
        assert "rollout_ep72" in final_state.subsystems

        assert harness.rm_handle.stop_cell.call_count >= 1
        assert harness.rm_handle.start_cell.call_count >= 1


class TestRolloutNumCellsValidation:
    def test_register_mismatched_num_cells_raises(self) -> None:
        """Declaring N rollout cells but registering M should fail."""
        harness = make_test_controller(rollout_num_cells=2)
        controller = harness.controller

        rm_handle = FakeRmHandle()
        cells = {"ep72": FakeRolloutCellAgent(cell_id="ep72", node_ids={"n1"})}
        rollout_agent = FakeRolloutAgent(cells=cells)

        with pytest.raises(AssertionError, match="Expected 2 rollout cells, got 1"):
            controller.register_rollout_subsystems(
                rm_handle=rm_handle,
                ft_rollout_agent=rollout_agent,
            )
