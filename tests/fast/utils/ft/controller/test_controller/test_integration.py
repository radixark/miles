"""Integration tests for M12: training + rollout subsystems in FtController."""

from __future__ import annotations

from typing import NamedTuple

import pytest
from prometheus_client import CollectorRegistry

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.controller import FtController, _RolloutSubsystemConfig, _build_rollout_subsystem_entry
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.factory import create_ft_controller
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.main.models import NormalState
from miles.utils.ft.controller.state_machines.subsystem import DetectingAnomaly, Recovering
from miles.utils.ft.controller.subsystem import MonitoringSustainedAliveConfig
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle
from tests.fast.utils.ft.conftest import (
    FakeMainJob,
    FakeNodeManager,
    FakeNotifier,
    FixedDecisionDetector,
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

    async def remote(self, *args: object, **kwargs: object) -> object:
        self.call_count += 1
        return self._result


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
        scrape_target_manager=metric_store,
        notifier=notifier,
        detectors=training_detectors,
        tick_interval=0.01,
        controller_exporter=controller_exporter,
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

    def test_rollout_entry_has_level1_restart(self) -> None:
        controller, *_ = _make_test_controller_with_rollout()
        state = controller._state_machine.state
        assert isinstance(state, NormalState)
        rollout = state.subsystems["rollout_ep72"]
        assert rollout.has_level1_restart is True

    def test_training_entry_has_no_level1_restart(self) -> None:
        controller, *_ = _make_test_controller_with_rollout()
        state = controller._state_machine.state
        assert isinstance(state, NormalState)
        training = state.subsystems["training"]
        assert training.has_level1_restart is False

    def test_rollout_configs_stored_on_controller(self) -> None:
        controller, *_ = _make_test_controller_with_rollout(cell_ids=["ep72", "ep36"])
        assert len(controller._rollout_configs) == 2
        cell_ids = [c.cell_id for c in controller._rollout_configs]
        assert "ep72" in cell_ids
        assert "ep36" in cell_ids

    def test_rollout_entry_returns_configured_node_ids(self) -> None:
        controller, *_ = _make_test_controller_with_rollout(
            cell_node_ids={"ep72": {"node-r0", "node-r1", "node-r2"}},
        )
        state = controller._state_machine.state
        assert isinstance(state, NormalState)
        rollout = state.subsystems["rollout_ep72"]
        assert rollout.get_active_node_ids() == {"node-r0", "node-r1", "node-r2"}


class TestNormalOperationWithRollout:
    @pytest.mark.anyio
    async def test_all_subsystems_healthy_stays_in_normal_state(self) -> None:
        """Training + rollout, no detectors fire → NormalState stays."""
        controller, *_ = _make_test_controller_with_rollout()

        for _ in range(3):
            await controller._tick()

        state = controller._state_machine.state
        assert isinstance(state, NormalState)
        assert "training" in state.subsystems
        assert "rollout_ep72" in state.subsystems

    @pytest.mark.anyio
    async def test_initial_subsystem_states_are_detecting_anomaly(self) -> None:
        controller, *_ = _make_test_controller_with_rollout()

        state = controller._state_machine.state
        assert isinstance(state, NormalState)

        for name, entry in state.subsystems.items():
            assert isinstance(entry.state_machine.state, DetectingAnomaly), (
                f"Expected DetectingAnomaly for {name}, got {type(entry.state_machine.state).__name__}"
            )


class TestRolloutCrashRecovery:
    @pytest.mark.anyio
    async def test_rollout_detector_fires_enters_recovery(self) -> None:
        """When a rollout detector fires, the rollout sub-SM enters Recovering."""
        controller, *_ = _make_test_controller_with_rollout()

        state = controller._state_machine.state
        assert isinstance(state, NormalState)

        rollout_entry = state.subsystems["rollout_ep72"]
        crash_detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["rollout-node-ep72-0"],
                reason="rollout cell ep72 dead",
                trigger=TriggerType.CRASH,
            )
        )
        rollout_entry.detectors.clear()
        rollout_entry.detectors.append(crash_detector)

        await controller._tick()

        rollout_state = rollout_entry.state_machine.state
        assert isinstance(rollout_state, Recovering), (
            f"Expected Recovering, got {type(rollout_state).__name__}"
        )


class TestCreateFreshSubsystemsIncludesRollout:
    @pytest.mark.anyio
    async def test_fresh_subsystems_include_rollout_after_registration(self) -> None:
        """After rollout registration, create_fresh_subsystems returns both training + rollout."""
        controller, *_ = _make_test_controller_with_rollout(
            cell_ids=["ep72", "ep36"],
        )

        fresh = controller._tick_loop._create_fresh_subsystems()
        assert "training" in fresh
        assert "rollout_ep72" in fresh
        assert "rollout_ep36" in fresh
        assert len(fresh) == 3

    @pytest.mark.anyio
    async def test_fresh_subsystems_have_detecting_anomaly_state(self) -> None:
        controller, *_ = _make_test_controller_with_rollout()

        fresh = controller._tick_loop._create_fresh_subsystems()
        for name, entry in fresh.items():
            assert isinstance(entry.state_machine.state, DetectingAnomaly), (
                f"Expected DetectingAnomaly for {name}, got {type(entry.state_machine.state).__name__}"
            )


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
        from tests.fast.utils.ft.conftest import make_test_controller

        harness = make_test_controller()
        status = harness.controller.get_status()
        assert status.rollout_subsystem_states is None


class TestBuildRolloutSubsystemEntry:
    def test_entry_has_correct_name_and_monitoring_config(self) -> None:
        rm_handle = FakeRmHandle()
        config = _RolloutSubsystemConfig(
            cell_id="ep72",
            rm_handle=rm_handle,
            get_active_node_ids=lambda: {"n1", "n2"},
        )
        entry = _build_rollout_subsystem_entry(name="rollout_ep72", config=config)

        assert entry.name == "rollout_ep72"
        assert entry.has_level1_restart is True
        assert entry.get_active_node_ids() == {"n1", "n2"}
        assert isinstance(entry.state_machine.state, DetectingAnomaly)
        assert isinstance(entry.monitoring_config, MonitoringSustainedAliveConfig)
        assert entry.monitoring_config.alive_duration_seconds == 180
