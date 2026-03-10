"""Tests for restart stepper handler classes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from tests.fast.utils.ft.utils.controller_fakes import (
    FakeNodeManager,
    FakeNotifier,
    FakeTrainingJob,
    failing_mark_node_bad,
    failing_stop_training,
    failing_submit_training,
)

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb

from miles.utils.ft.controller.state_machines.restart import (
    Evicting,
    MonitoringProgress,
    RestartContext,
    RestartDone,
    RestartFailed,
    StoppingAndRestarting,
    create_restart_stepper,
    iteration_progress,
)
from miles.utils.ft.utils.state_machine import StateMachineStepper


def _make_stepper() -> StateMachineStepper:
    return create_restart_stepper()


def _make_context(
    *,
    node_manager: FakeNodeManager | None = None,
    training_job: FakeTrainingJob | None = None,
    mini_wandb: MiniWandb | None = None,
    notifier: FakeNotifier | None = None,
    on_new_run: object | None = None,
    monitoring_success_iterations: int = 10,
    monitoring_timeout_seconds: int = 600,
    node_metadata: dict[str, dict[str, str]] | None = None,
) -> RestartContext:
    return RestartContext(
        node_manager=node_manager or FakeNodeManager(),
        training_job=training_job or FakeTrainingJob(),
        mini_wandb=mini_wandb or MiniWandb(),
        notifier=notifier,
        on_new_run=on_new_run,
        monitoring_success_iterations=monitoring_success_iterations,
        monitoring_timeout_seconds=monitoring_timeout_seconds,
        node_metadata=node_metadata or {},
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
        result = await stepper(state, ctx)

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
        result = await stepper(state, ctx)
        assert isinstance(result, StoppingAndRestarting)

    @pytest.mark.asyncio
    async def test_evicting_failure_returns_restart_failed(self) -> None:
        node_manager = FakeNodeManager()
        node_manager.mark_node_bad = failing_mark_node_bad  # type: ignore[assignment]
        stepper = _make_stepper()
        ctx = _make_context(node_manager=node_manager)

        state = Evicting(bad_node_ids=["node-A"])
        result = await stepper(state, ctx)
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
        result = await stepper(state, ctx)

        assert isinstance(result, StoppingAndRestarting)
        assert node_manager.is_node_bad("node-A")

    @pytest.mark.asyncio
    async def test_evicting_sends_notification(self) -> None:
        notifier = FakeNotifier()
        stepper = _make_stepper()
        ctx = _make_context(notifier=notifier)

        state = Evicting(bad_node_ids=["node-X"])
        await stepper(state, ctx)

        assert len(notifier.calls) == 1
        assert "Evicted" in notifier.calls[0][1]

    @pytest.mark.asyncio
    async def test_all_already_bad_still_transitions(self) -> None:
        """All nodes already marked bad -> still transitions to StoppingAndRestarting."""
        node_manager = FakeNodeManager()
        await node_manager.mark_node_bad("node-A", reason="prior")
        await node_manager.mark_node_bad("node-B", reason="prior")
        training_job = FakeTrainingJob()
        stepper = _make_stepper()
        ctx = _make_context(node_manager=node_manager, training_job=training_job)

        state = Evicting(bad_node_ids=["node-A", "node-B"])
        result = await stepper(state, ctx)

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
        result = await stepper(state, ctx)

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
        result = await stepper(state, ctx)

        assert isinstance(result, StoppingAndRestarting)
        assert node_manager.is_node_bad("ray-uuid-abc")
        assert node_manager.last_node_metadata is None


# ---------------------------------------------------------------------------
# StoppingAndRestarting
# ---------------------------------------------------------------------------


class TestStoppingAndRestarting:
    @pytest.mark.asyncio
    async def test_submit_phase_calls_stop_and_submit(self) -> None:
        training_job = FakeTrainingJob()
        stepper = _make_stepper()
        ctx = _make_context(training_job=training_job)

        state = StoppingAndRestarting(bad_node_ids=["node-A"])
        result = await stepper(state, ctx)

        assert isinstance(result, StoppingAndRestarting)
        assert result.submitted is True
        assert result.submit_time is not None
        assert training_job._stopped
        assert training_job._submitted

    @pytest.mark.asyncio
    async def test_submit_failure_returns_restart_failed(self) -> None:
        training_job = FakeTrainingJob()
        training_job.submit_training = failing_submit_training  # type: ignore[assignment]
        stepper = _make_stepper()
        ctx = _make_context(training_job=training_job)

        state = StoppingAndRestarting()
        result = await stepper(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.asyncio
    async def test_poll_running_transitions_to_monitoring(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=100, metrics={"iteration": 50})
        stepper = _make_stepper()
        ctx = _make_context(training_job=training_job, mini_wandb=mini_wandb)

        state = StoppingAndRestarting(submitted=True, submit_time=datetime.now(timezone.utc))
        result = await stepper(state, ctx)

        assert isinstance(result, MonitoringProgress)
        assert result.base_iteration == 50

    @pytest.mark.asyncio
    async def test_poll_failed_returns_restart_failed(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.FAILED])
        stepper = _make_stepper()
        ctx = _make_context(training_job=training_job)

        state = StoppingAndRestarting(submitted=True, submit_time=datetime.now(timezone.utc))
        result = await stepper(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.asyncio
    async def test_poll_pending_timeout_returns_restart_failed(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.PENDING])
        stepper = _make_stepper()
        ctx = _make_context(training_job=training_job)

        old_time = datetime.now(timezone.utc) - timedelta(seconds=400)
        state = StoppingAndRestarting(submitted=True, submit_time=old_time)
        result = await stepper(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.asyncio
    async def test_poll_pending_within_timeout_returns_none(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.PENDING])
        stepper = _make_stepper()
        ctx = _make_context(training_job=training_job)

        state = StoppingAndRestarting(submitted=True, submit_time=datetime.now(timezone.utc))
        result = await stepper(state, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_on_new_run_callback_called(self) -> None:
        captured_run_ids: list[str] = []
        training_job = FakeTrainingJob()
        stepper = _make_stepper()
        ctx = _make_context(training_job=training_job, on_new_run=captured_run_ids.append)

        state = StoppingAndRestarting(bad_node_ids=[])
        await stepper(state, ctx)
        assert len(captured_run_ids) == 1

    @pytest.mark.asyncio
    async def test_stop_training_exception_still_submits(self) -> None:
        """After stop_training fails but job is already stopped, submit_training proceeds."""
        training_job = FakeTrainingJob(status_sequence=[JobStatus.STOPPED])
        training_job.stop_training = failing_stop_training  # type: ignore[assignment]
        stepper = _make_stepper()
        ctx = _make_context(training_job=training_job)

        state = StoppingAndRestarting(bad_node_ids=["node-A"])
        result = await stepper(state, ctx)

        assert isinstance(result, StoppingAndRestarting)
        assert result.submitted is True
        assert training_job._submitted


# ---------------------------------------------------------------------------
# MonitoringProgress
# ---------------------------------------------------------------------------


class TestMonitoringProgress:
    @pytest.mark.asyncio
    async def test_monitoring_success(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=200, metrics={"iteration": 110})
        stepper = _make_stepper()
        ctx = _make_context(
            training_job=training_job,
            mini_wandb=mini_wandb,
            monitoring_success_iterations=10,
        )

        state = MonitoringProgress(
            bad_node_ids=["node-A"],
            start_time=datetime.now(timezone.utc),
            base_iteration=100,
        )
        result = await stepper(state, ctx)
        assert isinstance(result, RestartDone)
        assert result.bad_node_ids == ["node-A"]

    @pytest.mark.asyncio
    async def test_monitoring_job_failed(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.FAILED])
        stepper = _make_stepper()
        ctx = _make_context(training_job=training_job)

        state = MonitoringProgress(
            start_time=datetime.now(timezone.utc),
            base_iteration=0,
        )
        result = await stepper(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.asyncio
    async def test_monitoring_timeout(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        stepper = _make_stepper()
        ctx = _make_context(
            training_job=training_job,
            mini_wandb=mini_wandb,
            monitoring_timeout_seconds=60,
        )

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        state = MonitoringProgress(start_time=old_time, base_iteration=0)
        result = await stepper(state, ctx)
        assert isinstance(result, RestartFailed)

    @pytest.mark.asyncio
    async def test_monitoring_in_progress_returns_none(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 3})
        stepper = _make_stepper()
        ctx = _make_context(
            training_job=training_job,
            mini_wandb=mini_wandb,
            monitoring_success_iterations=10,
        )

        state = MonitoringProgress(
            start_time=datetime.now(timezone.utc),
            base_iteration=0,
        )
        result = await stepper(state, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_monitoring_timeout_transitions_to_failed(self) -> None:
        """Partial progress but timeout expired -> RestartFailed."""
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 5})
        stepper = _make_stepper()
        ctx = _make_context(
            training_job=training_job,
            mini_wandb=mini_wandb,
            monitoring_success_iterations=100,
            monitoring_timeout_seconds=60,
        )

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        state = MonitoringProgress(start_time=old_time, base_iteration=0)
        result = await stepper(state, ctx)
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


class TestTerminalStates:
    @pytest.mark.asyncio
    async def test_restart_done_is_terminal(self) -> None:
        stepper = _make_stepper()
        ctx = _make_context()
        result = await stepper(RestartDone(), ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_restart_failed_is_terminal(self) -> None:
        stepper = _make_stepper()
        ctx = _make_context()
        result = await stepper(RestartFailed(), ctx)
        assert result is None


# ---------------------------------------------------------------------------
# Full flow integration
# ---------------------------------------------------------------------------


class TestFullRestartFlow:
    @pytest.mark.asyncio
    async def test_evict_stop_monitor_done(self) -> None:
        """Evicting -> StoppingAndRestarting(submit) -> StoppingAndRestarting(poll) -> MonitoringProgress -> RestartDone."""
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 100})
        stepper = _make_stepper()
        ctx = _make_context(
            training_job=training_job,
            mini_wandb=mini_wandb,
            monitoring_success_iterations=5,
        )

        # Step 1: Evicting -> StoppingAndRestarting
        state = Evicting(bad_node_ids=["node-A"])
        state = await stepper(state, ctx)
        assert isinstance(state, StoppingAndRestarting)

        # Step 2: Submit
        state = await stepper(state, ctx)
        assert isinstance(state, StoppingAndRestarting)
        assert state.submitted

        # Step 3: Poll -> MonitoringProgress
        state = await stepper(state, ctx)
        assert isinstance(state, MonitoringProgress)
        assert state.base_iteration == 100

        # Step 4: progress achieved
        mini_wandb.log_step(run_id="r", step=2, metrics={"iteration": 110})
        state = await stepper(state, ctx)
        assert isinstance(state, RestartDone)

    @pytest.mark.asyncio
    async def test_direct_restart_no_eviction(self) -> None:
        """When bad_node_ids is empty, caller starts at StoppingAndRestarting."""
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 0})
        stepper = _make_stepper()
        ctx = _make_context(
            training_job=training_job,
            mini_wandb=mini_wandb,
            monitoring_success_iterations=5,
        )

        state = StoppingAndRestarting(bad_node_ids=[])
        state = await stepper(state, ctx)
        assert isinstance(state, StoppingAndRestarting)
        assert state.submitted

        state = await stepper(state, ctx)
        assert isinstance(state, MonitoringProgress)

        mini_wandb.log_step(run_id="r", step=2, metrics={"iteration": 10})
        state = await stepper(state, ctx)
        assert isinstance(state, RestartDone)
