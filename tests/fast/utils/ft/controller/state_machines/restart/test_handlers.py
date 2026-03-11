"""Tests for restart stepper handler classes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from tests.fast.utils.ft.utils.controller_fakes import (
    FakeMainJob,
    FakeNodeManager,
    FakeNotifier,
    failing_mark_node_bad,
    failing_stop_job,
    failing_submit_job,
)

from miles.utils.ft.adapters.types import JobStatus, SubsystemActuatorProtocol
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.restart import (
    Evicting,
    ExternalExecutionResult,
    MonitoringProgress,
    RestartContext,
    RestartDone,
    RestartFailed,
    RestartingMainJob,
    StoppingAndRestarting,
    create_restart_stepper,
    iteration_progress,
)
from miles.utils.ft.controller.subsystem import MonitoringIterationProgressConfig, MonitoringSustainedAliveConfig, RestartMode
from miles.utils.ft.utils.state_machine import StateMachineStepper


class FakeActuator(SubsystemActuatorProtocol):
    """In-memory actuator for subsystem restart tests."""

    def __init__(self, status_sequence: list[JobStatus] | None = None) -> None:
        self._status_sequence = status_sequence or [JobStatus.RUNNING]
        self._call_count: int = 0
        self.stopped: bool = False
        self.started: bool = False
        self.start_run_id: str = "actuator-run-1"

    async def stop(self) -> None:
        self.stopped = True

    async def start(self) -> str:
        self.started = True
        return self.start_run_id

    async def get_status(self) -> JobStatus:
        index = min(self._call_count, len(self._status_sequence) - 1)
        status = self._status_sequence[index]
        self._call_count += 1
        return status


def _make_stepper() -> StateMachineStepper:
    return create_restart_stepper()


def _make_context(
    *,
    node_manager: FakeNodeManager | None = None,
    main_job: FakeMainJob | None = None,
    mini_wandb: MiniWandb | None = None,
    notifier: FakeNotifier | None = None,
    on_new_run: object | None = None,
    node_metadata: dict[str, dict[str, str]] | None = None,
    actuator: SubsystemActuatorProtocol | None = None,
    monitoring_config: MonitoringIterationProgressConfig | MonitoringSustainedAliveConfig | None = None,
    restart_mode: RestartMode = RestartMode.SUBSYSTEM,
) -> RestartContext:
    resolved_main_job = main_job or FakeMainJob()
    return RestartContext(
        node_manager=node_manager or FakeNodeManager(),
        main_job=resolved_main_job,
        mini_wandb=mini_wandb or MiniWandb(),
        notifier=notifier,
        on_new_run=on_new_run,
        node_metadata=node_metadata or {},
        actuator=actuator or FakeActuator(),
        monitoring_config=monitoring_config or MonitoringIterationProgressConfig(),
        restart_mode=restart_mode,
    )


# ---------------------------------------------------------------------------
# Evicting
# ---------------------------------------------------------------------------


class TestEvicting:
    @pytest.mark.asyncio
    async def test_evicting_marks_bad_and_transitions_to_stopping(self) -> None:
        node_manager = FakeNodeManager()
        stepper = _make_stepper()
        ctx = _make_context(node_manager=node_manager)

        state = Evicting(bad_node_ids=["node-A"])
        result = await stepper.step_once(state, ctx)

        assert isinstance(result, StoppingAndRestarting)
        assert result.bad_node_ids == ["node-A"]
        assert node_manager.is_node_bad("node-A")

    @pytest.mark.asyncio
    async def test_evicting_skips_already_bad_nodes(self) -> None:
        node_manager = FakeNodeManager()
        await node_manager.mark_node_bad("node-A", reason="prior")
        stepper = _make_stepper()
        ctx = _make_context(node_manager=node_manager)

        state = Evicting(bad_node_ids=["node-A"])
        result = await stepper.step_once(state, ctx)
        assert isinstance(result, StoppingAndRestarting)

    @pytest.mark.asyncio
    async def test_evicting_failure_returns_restart_failed(self) -> None:
        node_manager = FakeNodeManager()
        node_manager.mark_node_bad = failing_mark_node_bad  # type: ignore[assignment]
        stepper = _make_stepper()
        ctx = _make_context(node_manager=node_manager)

        state = Evicting(bad_node_ids=["node-A"])
        result = await stepper.step_once(state, ctx)
        assert isinstance(result, RestartFailed)
        assert result.bad_node_ids == ["node-A"]

    @pytest.mark.asyncio
    async def test_get_bad_nodes_failure_continues_with_empty_set(self) -> None:
        """When K8s API (get_bad_nodes) fails, eviction continues (mark_bad is idempotent)."""
        node_manager = FakeNodeManager()

        async def _failing_get_bad_nodes() -> list[str]:
            raise ConnectionError("k8s API unavailable")

        node_manager.get_bad_nodes = _failing_get_bad_nodes  # type: ignore[assignment]
        stepper = _make_stepper()
        ctx = _make_context(node_manager=node_manager)

        state = Evicting(bad_node_ids=["node-A"])
        result = await stepper.step_once(state, ctx)

        assert isinstance(result, StoppingAndRestarting)
        assert node_manager.is_node_bad("node-A")

    @pytest.mark.asyncio
    async def test_evicting_sends_notification(self) -> None:
        notifier = FakeNotifier()
        stepper = _make_stepper()
        ctx = _make_context(notifier=notifier)

        state = Evicting(bad_node_ids=["node-X"])
        await stepper.step_once(state, ctx)

        assert len(notifier.calls) == 1
        assert "Evicted" in notifier.calls[0][1]

    @pytest.mark.asyncio
    async def test_all_already_bad_still_transitions(self) -> None:
        """All nodes already marked bad -> still transitions to StoppingAndRestarting."""
        node_manager = FakeNodeManager()
        await node_manager.mark_node_bad("node-A", reason="prior")
        await node_manager.mark_node_bad("node-B", reason="prior")
        main_job = FakeMainJob()
        stepper = _make_stepper()
        ctx = _make_context(node_manager=node_manager, main_job=main_job)

        state = Evicting(bad_node_ids=["node-A", "node-B"])
        result = await stepper.step_once(state, ctx)

        assert isinstance(result, StoppingAndRestarting)
        assert result.bad_node_ids == ["node-A", "node-B"]

    @pytest.mark.asyncio
    async def test_metadata_passed_to_mark_node_bad(self) -> None:
        """When node_metadata is set, mark_node_bad receives it for the matching node."""
        node_manager = FakeNodeManager()
        stepper = _make_stepper()
        metadata = {"ray-uuid-abc": {"k8s_node_name": "gke-node-01", "k8s_pod_name": "pod-x"}}
        ctx = _make_context(node_manager=node_manager, node_metadata=metadata)

        state = Evicting(bad_node_ids=["ray-uuid-abc"])
        result = await stepper.step_once(state, ctx)

        assert isinstance(result, StoppingAndRestarting)
        assert node_manager.is_node_bad("ray-uuid-abc")
        assert node_manager.last_node_metadata == {"k8s_node_name": "gke-node-01", "k8s_pod_name": "pod-x"}

    @pytest.mark.asyncio
    async def test_no_metadata_passes_none(self) -> None:
        """Without node_metadata for a node, mark_node_bad receives None (fallback)."""
        node_manager = FakeNodeManager()
        stepper = _make_stepper()
        ctx = _make_context(node_manager=node_manager)

        state = Evicting(bad_node_ids=["ray-uuid-abc"])
        result = await stepper.step_once(state, ctx)

        assert isinstance(result, StoppingAndRestarting)
        assert node_manager.is_node_bad("ray-uuid-abc")
        assert node_manager.last_node_metadata is None


# ---------------------------------------------------------------------------
# StoppingAndRestarting
# ---------------------------------------------------------------------------


class TestStoppingAndRestarting:
    @pytest.mark.asyncio
    async def test_main_job_restart_mode_returns_restarting_main_job(self) -> None:
        """restart_mode=MAIN_JOB -> RestartingMainJob(external_execution_result=None)."""
        stepper = _make_stepper()
        ctx = _make_context(restart_mode=RestartMode.MAIN_JOB)

        state = StoppingAndRestarting(bad_node_ids=["node-A"])
        result = await stepper.step_once(state, ctx)

        assert isinstance(result, RestartingMainJob)
        assert result.bad_node_ids == ["node-A"]
        assert result.external_execution_result is None

    @pytest.mark.asyncio
    async def test_subsystem_restart_submit_calls_actuator_stop_and_start(self) -> None:
        actuator = FakeActuator()
        stepper = _make_stepper()
        ctx = _make_context(actuator=actuator, restart_mode=RestartMode.SUBSYSTEM)

        state = StoppingAndRestarting(bad_node_ids=["node-A"])
        result = await stepper.step_once(state, ctx)

        assert isinstance(result, StoppingAndRestarting)
        assert result.submitted is True
        assert result.submit_time is not None
        assert actuator.stopped
        assert actuator.started

    @pytest.mark.asyncio
    async def test_subsystem_restart_actuator_stop_failure_returns_restart_failed(self) -> None:
        actuator = FakeActuator()
        actuator.stop = AsyncMock(side_effect=RuntimeError("stop failed"))  # type: ignore[assignment]
        stepper = _make_stepper()
        ctx = _make_context(actuator=actuator, restart_mode=RestartMode.SUBSYSTEM)

        state = StoppingAndRestarting()
        result = await stepper.step_once(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.asyncio
    async def test_subsystem_restart_actuator_start_failure_returns_restart_failed(self) -> None:
        actuator = FakeActuator()
        actuator.start = AsyncMock(side_effect=RuntimeError("start failed"))  # type: ignore[assignment]
        stepper = _make_stepper()
        ctx = _make_context(actuator=actuator, restart_mode=RestartMode.SUBSYSTEM)

        state = StoppingAndRestarting()
        result = await stepper.step_once(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.asyncio
    async def test_poll_running_transitions_to_monitoring(self) -> None:
        actuator = FakeActuator(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=100, metrics={"iteration": 50})
        stepper = _make_stepper()
        ctx = _make_context(actuator=actuator, mini_wandb=mini_wandb)

        state = StoppingAndRestarting(submitted=True, submit_time=datetime.now(timezone.utc))
        result = await stepper.step_once(state, ctx)

        assert isinstance(result, MonitoringProgress)
        assert result.base_iteration == 50

    @pytest.mark.asyncio
    async def test_poll_failed_returns_restart_failed(self) -> None:
        actuator = FakeActuator(status_sequence=[JobStatus.FAILED])
        stepper = _make_stepper()
        ctx = _make_context(actuator=actuator)

        state = StoppingAndRestarting(submitted=True, submit_time=datetime.now(timezone.utc))
        result = await stepper.step_once(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.asyncio
    async def test_poll_pending_timeout_returns_restart_failed(self) -> None:
        actuator = FakeActuator(status_sequence=[JobStatus.PENDING])
        stepper = _make_stepper()
        ctx = _make_context(actuator=actuator)

        old_time = datetime.now(timezone.utc) - timedelta(seconds=400)
        state = StoppingAndRestarting(submitted=True, submit_time=old_time)
        result = await stepper.step_once(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.asyncio
    async def test_poll_pending_within_timeout_returns_none(self) -> None:
        actuator = FakeActuator(status_sequence=[JobStatus.PENDING])
        stepper = _make_stepper()
        ctx = _make_context(actuator=actuator)

        state = StoppingAndRestarting(submitted=True, submit_time=datetime.now(timezone.utc))
        result = await stepper.step_once(state, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_on_new_run_callback_called(self) -> None:
        captured_run_ids: list[str] = []
        actuator = FakeActuator()
        stepper = _make_stepper()
        ctx = _make_context(actuator=actuator, on_new_run=captured_run_ids.append)

        state = StoppingAndRestarting(bad_node_ids=[])
        await stepper.step_once(state, ctx)
        assert len(captured_run_ids) == 1
        assert captured_run_ids[0] == actuator.start_run_id


# ---------------------------------------------------------------------------
# MonitoringProgress
# ---------------------------------------------------------------------------


class TestMonitoringProgress:
    @pytest.mark.asyncio
    async def test_monitoring_success(self) -> None:
        main_job = FakeMainJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=200, metrics={"iteration": 110})
        stepper = _make_stepper()
        ctx = _make_context(
            main_job=main_job,
            mini_wandb=mini_wandb,
            monitoring_config=MonitoringIterationProgressConfig(success_iterations=10),
        )

        state = MonitoringProgress(
            bad_node_ids=["node-A"],
            start_time=datetime.now(timezone.utc),
            base_iteration=100,
        )
        result = await stepper.step_once(state, ctx)
        assert isinstance(result, RestartDone)
        assert result.bad_node_ids == ["node-A"]

    @pytest.mark.asyncio
    async def test_monitoring_job_failed(self) -> None:
        main_job = FakeMainJob(status_sequence=[JobStatus.FAILED])
        stepper = _make_stepper()
        ctx = _make_context(main_job=main_job)

        state = MonitoringProgress(
            start_time=datetime.now(timezone.utc),
            base_iteration=0,
        )
        result = await stepper.step_once(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.asyncio
    async def test_monitoring_timeout(self) -> None:
        main_job = FakeMainJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        stepper = _make_stepper()
        ctx = _make_context(
            main_job=main_job,
            mini_wandb=mini_wandb,
            monitoring_config=MonitoringIterationProgressConfig(timeout_seconds=60),
        )

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        state = MonitoringProgress(start_time=old_time, base_iteration=0)
        result = await stepper.step_once(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.asyncio
    async def test_monitoring_in_progress_returns_none(self) -> None:
        main_job = FakeMainJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 3})
        stepper = _make_stepper()
        ctx = _make_context(
            main_job=main_job,
            mini_wandb=mini_wandb,
            monitoring_config=MonitoringIterationProgressConfig(success_iterations=10),
        )

        state = MonitoringProgress(
            start_time=datetime.now(timezone.utc),
            base_iteration=0,
        )
        result = await stepper.step_once(state, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_monitoring_timeout_transitions_to_failed(self) -> None:
        """Partial progress but timeout expired -> RestartFailed."""
        main_job = FakeMainJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 5})
        stepper = _make_stepper()
        ctx = _make_context(
            main_job=main_job,
            mini_wandb=mini_wandb,
            monitoring_config=MonitoringIterationProgressConfig(success_iterations=100, timeout_seconds=60),
        )

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        state = MonitoringProgress(start_time=old_time, base_iteration=0)
        result = await stepper.step_once(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.parametrize(
        "metric_value,expected_progress",
        [
            (float("nan"), 0),
            (float("inf"), 0),
            (float("-inf"), 0),
            (None, 0),
            (50, 0),
            (100, 0),
            (105, 5),
        ],
        ids=["nan", "inf", "neg_inf", "none", "negative_raw", "zero_raw", "normal"],
    )
    def test_iteration_progress_boundary_values(
        self,
        metric_value: float | None,
        expected_progress: int,
    ) -> None:
        mini_wandb = MiniWandb()
        if metric_value is not None:
            mini_wandb.set_active_run_id("r")
            mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": metric_value})

        state = MonitoringProgress(
            start_time=datetime.now(timezone.utc),
            base_iteration=100,
        )
        assert iteration_progress(state=state, mini_wandb=mini_wandb) == expected_progress


# ---------------------------------------------------------------------------
# Terminal states
# ---------------------------------------------------------------------------


class TestSustainedAlive:
    @pytest.mark.asyncio
    async def test_sustained_alive_success(self) -> None:
        """actuator.get_status()==RUNNING for alive_duration_seconds -> RestartDone."""
        actuator = FakeActuator(status_sequence=[JobStatus.RUNNING])
        stepper = _make_stepper()
        config = MonitoringSustainedAliveConfig(alive_duration_seconds=60, timeout_seconds=600)
        ctx = _make_context(
            actuator=actuator,
            monitoring_config=config,
        )

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        state = MonitoringProgress(start_time=old_time, base_iteration=0)
        result = await stepper.step_once(state, ctx)
        assert isinstance(result, RestartDone)

    @pytest.mark.asyncio
    async def test_sustained_alive_failed(self) -> None:
        """actuator.get_status()==FAILED -> RestartFailed."""
        actuator = FakeActuator(status_sequence=[JobStatus.FAILED])
        stepper = _make_stepper()
        config = MonitoringSustainedAliveConfig(alive_duration_seconds=60)
        ctx = _make_context(actuator=actuator, monitoring_config=config)

        state = MonitoringProgress(start_time=datetime.now(timezone.utc), base_iteration=0)
        result = await stepper.step_once(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.asyncio
    async def test_sustained_alive_timeout(self) -> None:
        """actuator.get_status()==PENDING past monitoring_timeout_seconds -> RestartFailed."""
        actuator = FakeActuator(status_sequence=[JobStatus.PENDING])
        stepper = _make_stepper()
        config = MonitoringSustainedAliveConfig(alive_duration_seconds=60, timeout_seconds=60)
        ctx = _make_context(
            actuator=actuator,
            monitoring_config=config,
        )

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        state = MonitoringProgress(start_time=old_time, base_iteration=0)
        result = await stepper.step_once(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.asyncio
    async def test_sustained_alive_in_progress_returns_none(self) -> None:
        """Running but not yet alive_duration_seconds -> None."""
        actuator = FakeActuator(status_sequence=[JobStatus.RUNNING])
        stepper = _make_stepper()
        config = MonitoringSustainedAliveConfig(alive_duration_seconds=300, timeout_seconds=600)
        ctx = _make_context(
            actuator=actuator,
            monitoring_config=config,
        )

        state = MonitoringProgress(start_time=datetime.now(timezone.utc), base_iteration=0)
        result = await stepper.step_once(state, ctx)
        assert result is None


# ---------------------------------------------------------------------------
# Terminal states
# ---------------------------------------------------------------------------


class TestTerminalStates:
    @pytest.mark.asyncio
    async def test_restart_done_is_terminal(self) -> None:
        stepper = _make_stepper()
        ctx = _make_context()
        result = await stepper.step_once(RestartDone(), ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_restart_failed_is_terminal(self) -> None:
        stepper = _make_stepper()
        ctx = _make_context()
        result = await stepper.step_once(RestartFailed(), ctx)
        assert result is None



# ---------------------------------------------------------------------------
# RestartingMainJob handler
# ---------------------------------------------------------------------------


class TestRestartingMainJob:
    @pytest.mark.asyncio
    async def test_no_result_returns_none(self) -> None:
        """external_execution_result=None -> handler returns None (waiting)."""
        stepper = _make_stepper()
        ctx = _make_context()

        state = RestartingMainJob(bad_node_ids=["node-A"], external_execution_result=None)
        result = await stepper.step_once(state, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_succeeded_returns_monitoring_progress(self) -> None:
        """SUCCEEDED -> handler returns MonitoringProgress."""
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 42})
        stepper = _make_stepper()
        ctx = _make_context(mini_wandb=mini_wandb)

        state = RestartingMainJob(
            bad_node_ids=["node-A"],
            external_execution_result=ExternalExecutionResult.SUCCEEDED,
        )
        result = await stepper.step_once(state, ctx)

        assert isinstance(result, MonitoringProgress)
        assert result.base_iteration == 42
        assert result.bad_node_ids == ["node-A"]

    @pytest.mark.asyncio
    async def test_succeeded_no_wandb_data_uses_zero_base(self) -> None:
        """SUCCEEDED with no wandb data -> base_iteration=0."""
        stepper = _make_stepper()
        ctx = _make_context()

        state = RestartingMainJob(external_execution_result=ExternalExecutionResult.SUCCEEDED)
        result = await stepper.step_once(state, ctx)

        assert isinstance(result, MonitoringProgress)
        assert result.base_iteration == 0

    @pytest.mark.asyncio
    async def test_failed_returns_restart_failed(self) -> None:
        """FAILED -> handler returns RestartFailed."""
        stepper = _make_stepper()
        ctx = _make_context()

        state = RestartingMainJob(
            bad_node_ids=["node-A"],
            external_execution_result=ExternalExecutionResult.FAILED,
        )
        result = await stepper.step_once(state, ctx)

        assert isinstance(result, RestartFailed)
        assert result.bad_node_ids == ["node-A"]

    @pytest.mark.asyncio
    async def test_timeout_returns_restart_failed(self) -> None:
        """TIMEOUT -> handler returns RestartFailed."""
        stepper = _make_stepper()
        ctx = _make_context()

        state = RestartingMainJob(
            bad_node_ids=["node-B"],
            external_execution_result=ExternalExecutionResult.TIMEOUT,
        )
        result = await stepper.step_once(state, ctx)

        assert isinstance(result, RestartFailed)
        assert result.bad_node_ids == ["node-B"]


# ---------------------------------------------------------------------------
# Full flow integration
# ---------------------------------------------------------------------------


class TestFullRestartFlow:
    @pytest.mark.asyncio
    async def test_subsystem_restart_evict_stop_monitor_done(self) -> None:
        """SUBSYSTEM: Evicting -> StoppingAndRestarting(submit) -> poll -> MonitoringProgress -> RestartDone."""
        actuator = FakeActuator(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 100})
        stepper = _make_stepper()
        ctx = _make_context(
            actuator=actuator,
            mini_wandb=mini_wandb,
            monitoring_config=MonitoringIterationProgressConfig(success_iterations=5),
            restart_mode=RestartMode.SUBSYSTEM,
        )

        # Step 1: Evicting -> StoppingAndRestarting
        state = Evicting(bad_node_ids=["node-A"])
        state = await stepper.step_once(state, ctx)
        assert isinstance(state, StoppingAndRestarting)

        # Step 2: Submit via actuator
        state = await stepper.step_once(state, ctx)
        assert isinstance(state, StoppingAndRestarting)
        assert state.submitted

        # Step 3: Poll -> MonitoringProgress
        state = await stepper.step_once(state, ctx)
        assert isinstance(state, MonitoringProgress)
        assert state.base_iteration == 100

        # Step 4: progress achieved
        mini_wandb.log_step(run_id="r", step=2, metrics={"iteration": 110})
        state = await stepper.step_once(state, ctx)
        assert isinstance(state, RestartDone)

    @pytest.mark.asyncio
    async def test_main_job_restart_evict_to_restarting_main_job(self) -> None:
        """MAIN_JOB: Evicting -> StoppingAndRestarting -> RestartingMainJob."""
        stepper = _make_stepper()
        ctx = _make_context(restart_mode=RestartMode.MAIN_JOB)

        # Step 1: Evicting -> StoppingAndRestarting
        state = Evicting(bad_node_ids=["node-A"])
        state = await stepper.step_once(state, ctx)
        assert isinstance(state, StoppingAndRestarting)

        # Step 2: Submit -> RestartingMainJob (MAIN_JOB mode, waiting for external result)
        state = await stepper.step_once(state, ctx)
        assert isinstance(state, RestartingMainJob)
        assert state.bad_node_ids == ["node-A"]
        assert state.external_execution_result is None

    @pytest.mark.asyncio
    async def test_subsystem_restart_direct_restart_no_eviction(self) -> None:
        """SUBSYSTEM: When bad_node_ids is empty, caller starts at StoppingAndRestarting."""
        actuator = FakeActuator(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 0})
        stepper = _make_stepper()
        ctx = _make_context(
            actuator=actuator,
            mini_wandb=mini_wandb,
            monitoring_config=MonitoringIterationProgressConfig(success_iterations=5),
            restart_mode=RestartMode.SUBSYSTEM,
        )

        state = StoppingAndRestarting(bad_node_ids=[])
        state = await stepper.step_once(state, ctx)
        assert isinstance(state, StoppingAndRestarting)
        assert state.submitted

        state = await stepper.step_once(state, ctx)
        assert isinstance(state, MonitoringProgress)

        mini_wandb.log_step(run_id="r", step=2, metrics={"iteration": 10})
        state = await stepper.step_once(state, ctx)
        assert isinstance(state, RestartDone)
