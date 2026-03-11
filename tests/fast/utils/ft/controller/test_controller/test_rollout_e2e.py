"""E2E tests for training + rollout FT behavior.

Uses real state machines, real detectors (GpuFaultDetector, RolloutCrashDetector,
etc.) with injected metrics via MiniPrometheus.  Mock MainJob, NodeManager,
Notifier, DiagnosticOrchestrator, and RmHandle provide controllable boundaries.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.detectors.chain import build_shared_hw_detectors
from miles.utils.ft.controller.detectors.core.rollout_crash import RolloutCrashDetector
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus
from miles.utils.ft.controller.state_machines.main.models import NormalSt
from miles.utils.ft.controller.state_machines.subsystem import DetectingAnomalySt
from miles.utils.ft.controller.subsystem import MonitoringSustainedAliveConfig
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle
from miles.utils.ft.agents.types import DiagnosticPipelineResult
from tests.fast.utils.ft.conftest import FakeDiagnosticOrchestrator
from tests.fast.utils.ft.controller.test_controller.test_integration import (
    _FakeRemoteMethod,
    _RolloutTestHarness,
    _make_test_controller_with_rollout,
)
from tests.fast.utils.ft.utils.metric_injectors import (
    inject_critical_xid,
    inject_gpu_unavailable,
    inject_healthy_node,
    inject_rollout_cell_alive,
)


def _override_rollout_monitoring(harness: _RolloutTestHarness, *, cell_ids: list[str] | None = None) -> None:
    """Set alive_duration_seconds=0 on rollout subsystem configs so tests don't wait 180s."""
    resolved = cell_ids or ["ep72"]
    for cell_id in resolved:
        config = harness.controller._tick_loop.subsystem_configs[f"rollout_{cell_id}"]
        config.monitoring_config = MonitoringSustainedAliveConfig(
            alive_duration_seconds=0,
            timeout_seconds=60,
        )


def _inject_crash_samples(
    store: MiniPrometheus,
    cell_id: str,
    *,
    span_seconds: float = 3.0,
    count: int = 20,
) -> None:
    """Inject rollout_cell_alive=False samples spanning *span_seconds*."""
    now = datetime.now(timezone.utc)
    for i in range(count):
        ts = now - timedelta(seconds=span_seconds * (1 - i / (count - 1)))
        inject_rollout_cell_alive(store, cell_id, alive=False, timestamp=ts)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Scenario 1: Rollout GPU XID -> recovery
# ---------------------------------------------------------------------------


class TestRolloutGpuXidRecovery:
    """GPU XID on a rollout-only node triggers rollout L1 recovery;
    training stays unaffected."""

    @pytest.mark.anyio
    async def test_rollout_gpu_xid_triggers_recovery_and_restores_normal(self) -> None:
        harness = _make_test_controller_with_rollout(
            training_detectors=build_shared_hw_detectors(),
            diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        )
        controller = harness.controller
        store = harness.metric_store
        _override_rollout_monitoring(harness)

        # Step 1: Inject healthy metrics for all nodes
        for node_id in ["train-node-0", "train-node-1", "rollout-0", "rollout-1"]:
            inject_healthy_node(store, node_id, num_gpus=8, num_nics=4)

        # Step 2: Tick several times -> all healthy
        for _ in range(3):
            await controller._tick()
        state = controller._state_machine.state
        assert isinstance(state, NormalSt)
        assert isinstance(state.subsystems["rollout_ep72"], DetectingAnomalySt)

        # Step 3: Inject GPU XID + unavailable on rollout-0
        inject_critical_xid(store, "rollout-0")
        inject_gpu_unavailable(store, "rollout-0", gpu="0")

        # Step 4: Tick -> rollout detects fault, cascades through L1 recovery
        await controller._tick()

        # Step 5: Verify
        state = controller._state_machine.state
        assert isinstance(state, NormalSt)
        assert isinstance(state.subsystems["rollout_ep72"], DetectingAnomalySt)
        assert isinstance(state.subsystems["training"], DetectingAnomalySt)

        assert harness.node_manager.was_ever_marked_bad("rollout-0")
        assert not harness.node_manager.was_ever_marked_bad("rollout-1")
        assert not harness.node_manager.was_ever_marked_bad("train-node-0")
        assert harness.rollout_manager_handle.stop_cell.call_count >= 1
        assert harness.rollout_manager_handle.start_cell.call_count >= 1
        assert not harness.main_job._stopped


# ---------------------------------------------------------------------------
# Scenario 2: Rollout engine crash (hardware OK)
# ---------------------------------------------------------------------------


class TestRolloutCrashRecovery:
    """RolloutCrashDetector fires when rollout_cell_alive stays 0 beyond
    threshold; L1 recovery restarts the cell while training is unaffected."""

    @pytest.mark.anyio
    async def test_rollout_crash_detector_triggers_recovery_when_cell_dead(self) -> None:
        harness = _make_test_controller_with_rollout(
            diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        )
        controller = harness.controller
        store = harness.metric_store

        rollout_config = controller._tick_loop.subsystem_configs["rollout_ep72"]
        rollout_config.detectors = [
            RolloutCrashDetector(cell_id="ep72", alive_threshold_seconds=2.0),
        ]
        _override_rollout_monitoring(harness)

        # Step 1: Inject healthy metrics + cell alive
        for node_id in ["rollout-0", "rollout-1"]:
            inject_healthy_node(store, node_id, num_gpus=8, num_nics=4)
        inject_rollout_cell_alive(store, "ep72", alive=True)

        for _ in range(3):
            await controller._tick()
        state = controller._state_machine.state
        assert isinstance(state, NormalSt)

        # Step 2: Inject sustained dead cell (spanning 3s > threshold=2s)
        _inject_crash_samples(store, "ep72", span_seconds=3.0)

        # Step 3: Tick -> crash detector fires -> recovery cascades
        await controller._tick()

        # Step 4: Verify
        state = controller._state_machine.state
        assert isinstance(state, NormalSt)
        assert isinstance(state.subsystems["rollout_ep72"], DetectingAnomalySt)
        assert isinstance(state.subsystems["training"], DetectingAnomalySt)

        assert harness.node_manager.was_ever_marked_bad("rollout-0")
        assert harness.node_manager.was_ever_marked_bad("rollout-1")
        assert harness.rollout_manager_handle.stop_cell.call_count >= 1
        assert harness.rollout_manager_handle.start_cell.call_count >= 1
        assert not harness.main_job._stopped


# ---------------------------------------------------------------------------
# Scenario 3: Co-located node fault
# ---------------------------------------------------------------------------


class TestColocatedNodeFault:
    """GPU XID on a node shared by training and rollout: training escalates
    to main-job restart (Level 2); rollout does L1 restart.  All subsystems
    are rebuilt after the main-job restart."""

    @pytest.mark.anyio
    async def test_shared_node_fault_triggers_main_job_restart_and_rebuilds_all(self) -> None:
        shared_node = "shared-node"
        harness = _make_test_controller_with_rollout(
            training_detectors=build_shared_hw_detectors(),
            monitoring_success_iterations=0,
            diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        )
        controller = harness.controller
        store = harness.metric_store

        controller.training_rank_roster.rank_placement[0] = shared_node
        controller.training_rank_roster.rank_placement[1] = "train-only-1"
        _override_rollout_monitoring(harness)

        # Step 1: Inject healthy metrics for all nodes
        for node_id in [shared_node, "train-only-1", "rollout-only-1"]:
            inject_healthy_node(store, node_id, num_gpus=8, num_nics=4)

        await controller._tick()
        state = controller._state_machine.state
        assert isinstance(state, NormalSt)

        # Step 2: Inject GPU XID on the shared node
        inject_critical_xid(store, shared_node)
        inject_gpu_unavailable(store, shared_node, gpu="0")

        # Step 3: Tick -> both subsystems detect, training L2 escalates to
        #   main-job restart; main SM rebuilds all subsystems
        await controller._tick()

        # Step 4: Verify
        state = controller._state_machine.state
        assert isinstance(state, NormalSt)
        assert "training" in state.subsystems
        assert "rollout_ep72" in state.subsystems

        for name, sub_state in state.subsystems.items():
            assert isinstance(sub_state, DetectingAnomalySt), (
                f"{name} not in DetectingAnomaly after main-job restart"
            )

        assert harness.node_manager.was_ever_marked_bad(shared_node)
        assert harness.main_job._stopped
        assert harness.main_job._submitted
        assert harness.rollout_manager_handle.stop_cell.call_count >= 1
        assert harness.rollout_manager_handle.start_cell.call_count >= 1


# ---------------------------------------------------------------------------
# Scenario 4: Multi-cell independent failures
# ---------------------------------------------------------------------------


class TestMultiCellIndependentFailures:
    """Two rollout cells fail at different times; each recovers independently
    without affecting training or the other cell."""

    @pytest.mark.anyio
    async def test_two_cells_recover_independently(self) -> None:
        harness = _make_test_controller_with_rollout(
            cell_ids=["ep72", "ep36"],
            diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        )
        controller = harness.controller
        store = harness.metric_store

        controller._tick_loop._cooldown = SlidingWindowThrottle(
            window_minutes=30.0, max_count=10,
        )

        for cell_id in ["ep72", "ep36"]:
            config = controller._tick_loop.subsystem_configs[f"rollout_{cell_id}"]
            config.detectors = [
                RolloutCrashDetector(cell_id=cell_id, alive_threshold_seconds=2.0),
            ]
        _override_rollout_monitoring(harness, cell_ids=["ep72", "ep36"])

        # Step 1: Inject healthy baselines
        for node_id in [
            "train-node-0", "train-node-1",
            "rollout-ep72-0", "rollout-ep72-1",
            "rollout-ep36-0", "rollout-ep36-1",
        ]:
            inject_healthy_node(store, node_id, num_gpus=8, num_nics=4)
        inject_rollout_cell_alive(store, "ep72", alive=True)
        inject_rollout_cell_alive(store, "ep36", alive=True)
        await controller._tick()

        # Step 2: Crash ep72 only
        _inject_crash_samples(store, "ep72", span_seconds=3.0)
        stop_before = harness.rollout_manager_handle.stop_cell.call_count
        stop_args_before = len(harness.rollout_manager_handle.stop_cell.call_args)

        # Step 3: Tick -> ep72 recovers, ep36 stays healthy
        await controller._tick()

        state = controller._state_machine.state
        assert isinstance(state, NormalSt)
        assert isinstance(state.subsystems["rollout_ep72"], DetectingAnomalySt)
        assert isinstance(state.subsystems["rollout_ep36"], DetectingAnomalySt)
        assert harness.rollout_manager_handle.stop_cell.call_count > stop_before

        ep72_stop_calls = [
            args for args in harness.rollout_manager_handle.stop_cell.call_args[stop_args_before:]
            if args[0] == "ep72"
        ]
        assert len(ep72_stop_calls) >= 1, "ep72 should have been targeted by stop_cell"

        # Step 4: Clear ep72 crash, crash ep36
        inject_rollout_cell_alive(store, "ep72", alive=True)
        _inject_crash_samples(store, "ep36", span_seconds=3.0)
        stop_before = harness.rollout_manager_handle.stop_cell.call_count
        stop_args_before = len(harness.rollout_manager_handle.stop_cell.call_args)

        # Step 5: Tick -> ep36 recovers
        await controller._tick()

        state = controller._state_machine.state
        assert isinstance(state, NormalSt)
        assert isinstance(state.subsystems["rollout_ep36"], DetectingAnomalySt)
        assert isinstance(state.subsystems["rollout_ep72"], DetectingAnomalySt)
        assert harness.rollout_manager_handle.stop_cell.call_count > stop_before

        ep36_stop_calls = [
            args for args in harness.rollout_manager_handle.stop_cell.call_args[stop_args_before:]
            if args[0] == "ep36"
        ]
        assert len(ep36_stop_calls) >= 1, "ep36 should have been targeted by stop_cell"

        assert not harness.main_job._stopped


# ---------------------------------------------------------------------------
# Scenario 5: L1 failure -> diagnostics -> NotifyHumans
# ---------------------------------------------------------------------------


class TestRolloutLevel1FailureNotifyHumans:
    """L1 restart repeatedly returns FAILED status -> RestartFailed ->
    StopTimeDiagnostics -> diagnostics find bad nodes ->
    EvictingAndRestarting(final=True) -> still FAILED -> NotifyHumans.
    Notifier receives a Recovery Alert and rollout returns to DetectingAnomaly."""

    @pytest.mark.anyio
    async def test_level1_failure_runs_diagnostics_then_notifies_humans(self) -> None:
        diag_orch = FakeDiagnosticOrchestrator(
            result=DiagnosticPipelineResult(
                bad_node_ids=["rollout-0"],
                reason="fake diagnostic found bad node",
            ),
        )
        harness = _make_test_controller_with_rollout(
            diagnostic_orchestrator=diag_orch,
        )
        controller = harness.controller
        store = harness.metric_store

        rollout_config = controller._tick_loop.subsystem_configs["rollout_ep72"]
        rollout_config.detectors = [
            RolloutCrashDetector(cell_id="ep72", alive_threshold_seconds=2.0),
        ]
        _override_rollout_monitoring(harness)

        harness.rollout_manager_handle.get_cell_status = _FakeRemoteMethod(result=JobStatus.FAILED)

        # Step 1: Inject healthy metrics + cell alive
        for node_id in ["rollout-0", "rollout-1"]:
            inject_healthy_node(store, node_id, num_gpus=8, num_nics=4)
        inject_rollout_cell_alive(store, "ep72", alive=True)
        await controller._tick()

        # Step 2: Inject sustained dead cell
        _inject_crash_samples(store, "ep72", span_seconds=3.0)

        # Step 3: Tick -> crash -> L1 restart -> FAILED -> StopTimeDiagnostics
        #   -> diagnostics find bad_node_ids -> EvictingAndRestarting(final=True)
        #   -> Evicting -> StoppingAndRestarting -> still FAILED -> NotifyHumans
        await controller._tick()

        # Step 4: Verify
        state = controller._state_machine.state
        assert isinstance(state, NormalSt)
        assert isinstance(state.subsystems["rollout_ep72"], DetectingAnomalySt)

        assert harness.rollout_manager_handle.stop_cell.call_count >= 2, (
            f"stop_cell should be called at least twice (first L1 + final retry), got {harness.rollout_manager_handle.stop_cell.call_count}"
        )
        assert harness.rollout_manager_handle.start_cell.call_count >= 2, (
            f"start_cell should be called at least twice (first L1 + final retry), got {harness.rollout_manager_handle.start_cell.call_count}"
        )
        assert diag_orch.call_count >= 1

        recovery_alerts = [
            (title, content)
            for title, content, _ in harness.notifier.calls
            if title == "Recovery Alert"
        ]
        assert len(recovery_alerts) >= 1, (
            f"Expected 'Recovery Alert' notification, got: {harness.notifier.calls}"
        )

        assert not harness.main_job._stopped
