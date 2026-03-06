"""Tests for recovery phase handler functions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery_orchestrator.context import (
    PENDING_TIMEOUT_SECONDS,
    RecoveryContext,
)
from miles.utils.ft.controller.recovery_orchestrator.phase_handlers import (
    _iteration_progress,
    _reattempt_poll,
    step_check_alerts,
    step_diagnosing,
    step_evict_and_restart,
    step_monitoring,
    step_notify,
    step_reattempting,
)
from miles.utils.ft.models._fault import ActionType, Decision, TriggerType
from miles.utils.ft.models._recovery import RecoveryPhase
from miles.utils.ft.protocols.platform import JobStatus
from tests.fast.utils.ft.conftest import (
    FakeDiagnosticScheduler,
    FakeNodeManager,
    FakeNotifier,
    FakeTrainingJob,
)


def _make_mini_wandb_with_iteration(value: float | None) -> MiniWandb:
    mini_wandb = MiniWandb()
    if value is not None:
        mini_wandb.log_step(run_id="test", step=1, metrics={"iteration": value})
    return mini_wandb


def _make_ctx(**overrides: object) -> RecoveryContext:
    defaults: dict = dict(trigger=TriggerType.CRASH)
    defaults.update(overrides)
    return RecoveryContext(**defaults)


# ===================================================================
# step_check_alerts
# ===================================================================


class _FakeAlertChecker:
    def __init__(self, bad_node_ids: list[str], reasons: list[str]) -> None:
        self._bad_node_ids = bad_node_ids
        self._reasons = reasons

    def check_alerts(self) -> tuple[list[str], list[str]]:
        return self._bad_node_ids, self._reasons


class TestStepCheckAlerts:
    @pytest.mark.anyio
    async def test_bad_nodes_found_transitions_to_evict(self) -> None:
        ctx = _make_ctx()
        checker = _FakeAlertChecker(bad_node_ids=["node-A"], reasons=["gpu lost"])

        result = await step_check_alerts(ctx, checker)

        assert result == RecoveryPhase.EVICT_AND_RESTART
        assert ctx.bad_node_ids == ["node-A"]

    @pytest.mark.anyio
    async def test_no_alerts_transitions_to_reattempting(self) -> None:
        ctx = _make_ctx()
        checker = _FakeAlertChecker(bad_node_ids=[], reasons=[])

        result = await step_check_alerts(ctx, checker)

        assert result == RecoveryPhase.REATTEMPTING


# ===================================================================
# step_reattempting / _reattempt_submit / _reattempt_poll
# ===================================================================


class TestStepReattempting:
    @pytest.mark.anyio
    async def test_not_submitted_delegates_to_submit(self) -> None:
        """When reattempt_submitted=False, step_reattempting calls _reattempt_submit."""
        ctx = _make_ctx()
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()

        result = await step_reattempting(ctx, training_job, mini_wandb)

        assert ctx.reattempt_submitted is True
        assert result is None

    @pytest.mark.anyio
    async def test_already_submitted_delegates_to_poll(self) -> None:
        """When reattempt_submitted=True, step_reattempting calls _reattempt_poll."""
        ctx = _make_ctx()
        ctx.reattempt_submitted = True
        ctx.reattempt_submit_time = datetime.now(timezone.utc)
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()

        result = await step_reattempting(ctx, training_job, mini_wandb)

        assert result == RecoveryPhase.MONITORING

    @pytest.mark.anyio
    async def test_submit_failure_transitions_to_notify(self) -> None:
        training_job = FakeTrainingJob()

        async def failing_submit(excluded_node_ids: list[str] | None = None) -> str:
            raise RuntimeError("submit failed")

        training_job.submit_training = failing_submit  # type: ignore[assignment]
        ctx = _make_ctx()
        mini_wandb = MiniWandb()

        result = await step_reattempting(ctx, training_job, mini_wandb)

        assert result == RecoveryPhase.NOTIFY
        assert ctx.reattempt_submitted is False


class TestReattemptPollBranches:
    @pytest.mark.anyio
    async def test_running_transitions_to_monitoring(self) -> None:
        ctx = _make_ctx()
        ctx.reattempt_submitted = True
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = _make_mini_wandb_with_iteration(42.0)

        result = await _reattempt_poll(ctx, training_job, mini_wandb)

        assert result == RecoveryPhase.MONITORING
        assert ctx.reattempt_base_iteration == 42
        assert ctx.reattempt_start_time is not None

    @pytest.mark.anyio
    async def test_failed_transitions_to_diagnosing(self) -> None:
        ctx = _make_ctx()
        ctx.reattempt_submitted = True
        training_job = FakeTrainingJob(status_sequence=[JobStatus.FAILED])
        mini_wandb = MiniWandb()

        result = await _reattempt_poll(ctx, training_job, mini_wandb)

        assert result == RecoveryPhase.DIAGNOSING

    @pytest.mark.anyio
    async def test_pending_timeout_transitions_to_notify(self) -> None:
        ctx = _make_ctx()
        ctx.reattempt_submitted = True
        ctx.reattempt_submit_time = datetime.now(timezone.utc) - timedelta(
            seconds=PENDING_TIMEOUT_SECONDS + 10,
        )
        training_job = FakeTrainingJob(status_sequence=[JobStatus.PENDING])
        mini_wandb = MiniWandb()

        result = await _reattempt_poll(ctx, training_job, mini_wandb)

        assert result == RecoveryPhase.NOTIFY

    @pytest.mark.anyio
    async def test_still_pending_returns_none(self) -> None:
        ctx = _make_ctx()
        ctx.reattempt_submitted = True
        ctx.reattempt_submit_time = datetime.now(timezone.utc)
        training_job = FakeTrainingJob(status_sequence=[JobStatus.PENDING])
        mini_wandb = MiniWandb()

        result = await _reattempt_poll(ctx, training_job, mini_wandb)

        assert result is None

    @pytest.mark.anyio
    async def test_running_with_nan_iteration_sets_base_zero(self) -> None:
        ctx = _make_ctx()
        ctx.reattempt_submitted = True
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = _make_mini_wandb_with_iteration(float("nan"))

        result = await _reattempt_poll(ctx, training_job, mini_wandb)

        assert result is not None
        assert ctx.reattempt_base_iteration == 0

    @pytest.mark.anyio
    async def test_running_with_inf_iteration_sets_base_zero(self) -> None:
        ctx = _make_ctx()
        ctx.reattempt_submitted = True
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = _make_mini_wandb_with_iteration(float("inf"))

        result = await _reattempt_poll(ctx, training_job, mini_wandb)

        assert result is not None
        assert ctx.reattempt_base_iteration == 0

    @pytest.mark.anyio
    async def test_running_with_none_iteration_sets_base_zero(self) -> None:
        ctx = _make_ctx()
        ctx.reattempt_submitted = True
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()

        result = await _reattempt_poll(ctx, training_job, mini_wandb)

        assert result is not None
        assert ctx.reattempt_base_iteration == 0


# ===================================================================
# step_monitoring
# ===================================================================


class TestStepMonitoring:
    @pytest.mark.anyio
    async def test_failed_transitions_to_diagnosing(self) -> None:
        ctx = _make_ctx(monitoring_success_iterations=5, monitoring_timeout_seconds=600)
        ctx.reattempt_start_time = datetime.now(timezone.utc)
        ctx.reattempt_base_iteration = 0
        training_job = FakeTrainingJob(status_sequence=[JobStatus.FAILED])
        mini_wandb = _make_mini_wandb_with_iteration(3.0)

        result = await step_monitoring(ctx, training_job, mini_wandb)

        assert result == RecoveryPhase.DIAGNOSING

    @pytest.mark.anyio
    async def test_monitoring_timeout_transitions_to_diagnosing(self) -> None:
        ctx = _make_ctx(monitoring_success_iterations=100, monitoring_timeout_seconds=60)
        ctx.reattempt_start_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        ctx.reattempt_base_iteration = 0
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = _make_mini_wandb_with_iteration(5.0)

        result = await step_monitoring(ctx, training_job, mini_wandb)

        assert result == RecoveryPhase.DIAGNOSING

    @pytest.mark.anyio
    async def test_exact_threshold_triggers_done(self) -> None:
        ctx = _make_ctx(monitoring_success_iterations=5, monitoring_timeout_seconds=600)
        ctx.reattempt_start_time = datetime.now(timezone.utc)
        ctx.reattempt_base_iteration = 0
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = _make_mini_wandb_with_iteration(5.0)

        result = await step_monitoring(ctx, training_job, mini_wandb)

        assert result == RecoveryPhase.DONE

    @pytest.mark.anyio
    async def test_one_below_threshold_waits(self) -> None:
        ctx = _make_ctx(monitoring_success_iterations=5, monitoring_timeout_seconds=600)
        ctx.reattempt_start_time = datetime.now(timezone.utc)
        ctx.reattempt_base_iteration = 0
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = _make_mini_wandb_with_iteration(4.0)

        result = await step_monitoring(ctx, training_job, mini_wandb)

        assert result is None

    @pytest.mark.anyio
    async def test_no_reattempt_start_time_skips_timeout_check(self) -> None:
        """When reattempt_start_time is None, timeout check is skipped."""
        ctx = _make_ctx(monitoring_success_iterations=100, monitoring_timeout_seconds=1)
        ctx.reattempt_start_time = None
        ctx.reattempt_base_iteration = 0
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = _make_mini_wandb_with_iteration(5.0)

        result = await step_monitoring(ctx, training_job, mini_wandb)

        assert result is None


# ===================================================================
# step_diagnosing
# ===================================================================


class TestStepDiagnosing:
    @pytest.mark.anyio
    async def test_bad_nodes_found_transitions_to_evict(self) -> None:
        decision = Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=["node-X"],
            reason="gpu diagnostic failed",
        )
        scheduler = FakeDiagnosticScheduler(decision=decision)
        ctx = _make_ctx()

        result = await step_diagnosing(ctx, scheduler)

        assert result == RecoveryPhase.EVICT_AND_RESTART
        assert ctx.bad_node_ids == ["node-X"]

    @pytest.mark.anyio
    async def test_all_passed_transitions_to_notify(self) -> None:
        decision = Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="all diagnostics passed",
        )
        scheduler = FakeDiagnosticScheduler(decision=decision)
        ctx = _make_ctx()

        result = await step_diagnosing(ctx, scheduler)

        assert result == RecoveryPhase.NOTIFY


# ===================================================================
# step_evict_and_restart
# ===================================================================


class TestStepEvictAndRestart:
    @pytest.mark.anyio
    async def test_empty_bad_nodes_transitions_to_notify(self) -> None:
        ctx = _make_ctx()
        ctx.bad_node_ids = []

        result = await step_evict_and_restart(
            ctx,
            node_manager=FakeNodeManager(),
            training_job=FakeTrainingJob(),
            mini_wandb=MiniWandb(),
        )

        assert result == RecoveryPhase.NOTIFY

    @pytest.mark.anyio
    async def test_successful_eviction_and_restart(self) -> None:
        ctx = _make_ctx()
        ctx.bad_node_ids = ["node-bad"]
        node_manager = FakeNodeManager()
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()

        result = await step_evict_and_restart(
            ctx,
            node_manager=node_manager,
            training_job=training_job,
            mini_wandb=mini_wandb,
        )

        assert result == RecoveryPhase.DONE
        assert node_manager.is_node_bad("node-bad")
        assert training_job._submitted

    @pytest.mark.anyio
    async def test_eviction_failure_transitions_to_notify(self) -> None:
        ctx = _make_ctx()
        ctx.bad_node_ids = ["node-bad"]
        node_manager = FakeNodeManager()

        async def always_fail(node_id: str, reason: str = "") -> None:
            raise RuntimeError("eviction failed")

        node_manager.mark_node_bad = always_fail  # type: ignore[assignment]
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()

        result = await step_evict_and_restart(
            ctx,
            node_manager=node_manager,
            training_job=training_job,
            mini_wandb=mini_wandb,
        )

        assert result == RecoveryPhase.NOTIFY

    @pytest.mark.anyio
    async def test_restart_failure_transitions_to_notify(self) -> None:
        ctx = _make_ctx()
        ctx.bad_node_ids = ["node-bad"]
        node_manager = FakeNodeManager()
        training_job = FakeTrainingJob()

        async def failing_submit(excluded_node_ids: list[str] | None = None) -> str:
            raise RuntimeError("submit failed")

        training_job.submit_training = failing_submit  # type: ignore[assignment]
        mini_wandb = MiniWandb()

        result = await step_evict_and_restart(
            ctx,
            node_manager=node_manager,
            training_job=training_job,
            mini_wandb=mini_wandb,
        )

        assert result == RecoveryPhase.NOTIFY
        assert node_manager.is_node_bad("node-bad")

    @pytest.mark.anyio
    async def test_multiple_bad_nodes_all_evicted(self) -> None:
        ctx = _make_ctx()
        ctx.bad_node_ids = ["node-A", "node-B"]
        node_manager = FakeNodeManager()
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()

        result = await step_evict_and_restart(
            ctx,
            node_manager=node_manager,
            training_job=training_job,
            mini_wandb=mini_wandb,
        )

        assert result == RecoveryPhase.DONE
        assert node_manager.is_node_bad("node-A")
        assert node_manager.is_node_bad("node-B")


# ===================================================================
# step_notify
# ===================================================================


class TestStepNotify:
    @pytest.mark.anyio
    async def test_sends_notification_and_returns_done(self) -> None:
        ctx = _make_ctx()
        ctx.phase_before_notify = RecoveryPhase.DIAGNOSING
        notifier = FakeNotifier()

        result = await step_notify(ctx, notifier)

        assert result == RecoveryPhase.DONE
        assert len(notifier.calls) == 1
        title, content, _ = notifier.calls[0]
        assert title == "Recovery Alert"
        assert "diagnosing" in content

    @pytest.mark.anyio
    async def test_none_notifier_does_not_crash(self) -> None:
        ctx = _make_ctx()
        ctx.phase_before_notify = RecoveryPhase.EVICT_AND_RESTART

        result = await step_notify(ctx, notifier=None)

        assert result == RecoveryPhase.DONE

    @pytest.mark.anyio
    async def test_unknown_phase_before_notify(self) -> None:
        ctx = _make_ctx()
        ctx.phase_before_notify = None
        notifier = FakeNotifier()

        result = await step_notify(ctx, notifier)

        assert result == RecoveryPhase.DONE
        assert "unknown" in notifier.calls[0][1]


# ===================================================================
# _iteration_progress
# ===================================================================


class TestIterationProgress:
    def test_none_iteration_returns_zero(self) -> None:
        ctx = _make_ctx(reattempt_base_iteration=5)
        mini_wandb = MiniWandb()
        assert _iteration_progress(ctx, mini_wandb) == 0

    def test_nan_iteration_returns_zero(self) -> None:
        ctx = _make_ctx(reattempt_base_iteration=5)
        mini_wandb = _make_mini_wandb_with_iteration(float("nan"))
        assert _iteration_progress(ctx, mini_wandb) == 0

    def test_positive_inf_iteration_returns_zero(self) -> None:
        ctx = _make_ctx(reattempt_base_iteration=5)
        mini_wandb = _make_mini_wandb_with_iteration(float("inf"))
        assert _iteration_progress(ctx, mini_wandb) == 0

    def test_negative_inf_iteration_returns_zero(self) -> None:
        ctx = _make_ctx(reattempt_base_iteration=5)
        mini_wandb = _make_mini_wandb_with_iteration(float("-inf"))
        assert _iteration_progress(ctx, mini_wandb) == 0

    def test_negative_progress_returns_zero(self) -> None:
        """Simulates a run reset where current_iteration < base_iteration."""
        ctx = _make_ctx(reattempt_base_iteration=100)
        mini_wandb = _make_mini_wandb_with_iteration(10.0)
        assert _iteration_progress(ctx, mini_wandb) == 0

    def test_none_base_iteration_treated_as_zero(self) -> None:
        ctx = _make_ctx(reattempt_base_iteration=None)
        mini_wandb = _make_mini_wandb_with_iteration(7.0)
        assert _iteration_progress(ctx, mini_wandb) == 7

    def test_normal_progress(self) -> None:
        ctx = _make_ctx(reattempt_base_iteration=10)
        mini_wandb = _make_mini_wandb_with_iteration(15.0)
        assert _iteration_progress(ctx, mini_wandb) == 5

    def test_zero_progress(self) -> None:
        ctx = _make_ctx(reattempt_base_iteration=10)
        mini_wandb = _make_mini_wandb_with_iteration(10.0)
        assert _iteration_progress(ctx, mini_wandb) == 0
