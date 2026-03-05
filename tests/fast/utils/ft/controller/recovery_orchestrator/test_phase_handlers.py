"""Tests for phase_handlers edge cases: _iteration_progress and _reattempt_poll boundaries."""
from __future__ import annotations

import math
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
    step_monitoring,
)
from miles.utils.ft.protocols.platform import JobStatus
from tests.fast.utils.ft.conftest import FakeTrainingJob


def _make_mini_wandb_with_iteration(value: float | None) -> MiniWandb:
    mini_wandb = MiniWandb()
    if value is not None:
        mini_wandb.log_step(run_id="test", step=1, metrics={"iteration": value})
    return mini_wandb


class TestIterationProgress:
    def test_none_iteration_returns_zero(self) -> None:
        ctx = RecoveryContext(trigger="crash", reattempt_base_iteration=5)
        mini_wandb = MiniWandb()
        assert _iteration_progress(ctx, mini_wandb) == 0

    def test_nan_iteration_returns_zero(self) -> None:
        ctx = RecoveryContext(trigger="crash", reattempt_base_iteration=5)
        mini_wandb = _make_mini_wandb_with_iteration(float("nan"))
        assert _iteration_progress(ctx, mini_wandb) == 0

    def test_positive_inf_iteration_returns_zero(self) -> None:
        ctx = RecoveryContext(trigger="crash", reattempt_base_iteration=5)
        mini_wandb = _make_mini_wandb_with_iteration(float("inf"))
        assert _iteration_progress(ctx, mini_wandb) == 0

    def test_negative_inf_iteration_returns_zero(self) -> None:
        ctx = RecoveryContext(trigger="crash", reattempt_base_iteration=5)
        mini_wandb = _make_mini_wandb_with_iteration(float("-inf"))
        assert _iteration_progress(ctx, mini_wandb) == 0

    def test_negative_progress_returns_zero(self) -> None:
        """Simulates a run reset where current_iteration < base_iteration."""
        ctx = RecoveryContext(trigger="crash", reattempt_base_iteration=100)
        mini_wandb = _make_mini_wandb_with_iteration(10.0)
        assert _iteration_progress(ctx, mini_wandb) == 0

    def test_none_base_iteration_treated_as_zero(self) -> None:
        ctx = RecoveryContext(trigger="crash", reattempt_base_iteration=None)
        mini_wandb = _make_mini_wandb_with_iteration(7.0)
        assert _iteration_progress(ctx, mini_wandb) == 7

    def test_normal_progress(self) -> None:
        ctx = RecoveryContext(trigger="crash", reattempt_base_iteration=10)
        mini_wandb = _make_mini_wandb_with_iteration(15.0)
        assert _iteration_progress(ctx, mini_wandb) == 5

    def test_zero_progress(self) -> None:
        ctx = RecoveryContext(trigger="crash", reattempt_base_iteration=10)
        mini_wandb = _make_mini_wandb_with_iteration(10.0)
        assert _iteration_progress(ctx, mini_wandb) == 0


class TestReattemptPollNaNIteration:
    """Verify _reattempt_poll sets reattempt_base_iteration=0 when iteration is NaN/Inf."""

    @pytest.mark.anyio
    async def test_running_with_nan_iteration_sets_base_zero(self) -> None:
        ctx = RecoveryContext(trigger="crash")
        ctx.reattempt_submitted = True
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = _make_mini_wandb_with_iteration(float("nan"))

        result = await _reattempt_poll(ctx, training_job, mini_wandb)

        assert result is not None
        assert ctx.reattempt_base_iteration == 0

    @pytest.mark.anyio
    async def test_running_with_inf_iteration_sets_base_zero(self) -> None:
        ctx = RecoveryContext(trigger="crash")
        ctx.reattempt_submitted = True
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = _make_mini_wandb_with_iteration(float("inf"))

        result = await _reattempt_poll(ctx, training_job, mini_wandb)

        assert result is not None
        assert ctx.reattempt_base_iteration == 0

    @pytest.mark.anyio
    async def test_running_with_none_iteration_sets_base_zero(self) -> None:
        ctx = RecoveryContext(trigger="crash")
        ctx.reattempt_submitted = True
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()

        result = await _reattempt_poll(ctx, training_job, mini_wandb)

        assert result is not None
        assert ctx.reattempt_base_iteration == 0

    @pytest.mark.anyio
    async def test_running_with_valid_iteration_sets_base(self) -> None:
        ctx = RecoveryContext(trigger="crash")
        ctx.reattempt_submitted = True
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = _make_mini_wandb_with_iteration(42.0)

        result = await _reattempt_poll(ctx, training_job, mini_wandb)

        assert result is not None
        assert ctx.reattempt_base_iteration == 42


class TestMonitoringSuccessExactThreshold:
    """Verify step_monitoring transitions to DONE at exactly monitoring_success_iterations."""

    @pytest.mark.anyio
    async def test_exact_threshold_triggers_done(self) -> None:
        ctx = RecoveryContext(
            trigger="crash",
            monitoring_success_iterations=5,
            monitoring_timeout_seconds=600,
        )
        ctx.reattempt_start_time = datetime.now(timezone.utc)
        ctx.reattempt_base_iteration = 0
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = _make_mini_wandb_with_iteration(5.0)

        result = await step_monitoring(ctx, training_job, mini_wandb)
        assert result is not None

    @pytest.mark.anyio
    async def test_one_below_threshold_waits(self) -> None:
        ctx = RecoveryContext(
            trigger="crash",
            monitoring_success_iterations=5,
            monitoring_timeout_seconds=600,
        )
        ctx.reattempt_start_time = datetime.now(timezone.utc)
        ctx.reattempt_base_iteration = 0
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = _make_mini_wandb_with_iteration(4.0)

        result = await step_monitoring(ctx, training_job, mini_wandb)
        assert result is None
