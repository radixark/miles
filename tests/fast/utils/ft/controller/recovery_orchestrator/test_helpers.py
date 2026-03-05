"""Tests for recovery_orchestrator/helpers.py (retry_async, stop_clear_submit)."""
from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery_orchestrator.helpers import (
    RetryResult,
    retry_async,
    stop_clear_submit,
)
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import FakeTrainingJob


class TestRetryAsyncEdgePaths:
    def test_succeeds_on_second_attempt(self) -> None:
        call_count = 0

        async def flaky_fn() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")

        result = asyncio.run(retry_async(func=flaky_fn, description="test_retry"))

        assert isinstance(result, RetryResult)
        assert result.ok is True
        assert call_count == 2

    def test_all_retries_fail_returns_failure(self) -> None:
        async def always_fail() -> None:
            raise RuntimeError("permanent error")

        result = asyncio.run(retry_async(
            func=always_fail, description="test_fail", max_retries=2,
        ))

        assert isinstance(result, RetryResult)
        assert result.ok is False
        assert result.error is not None
        assert "permanent error" in result.error

    def test_preserves_return_value(self) -> None:
        async def returns_value() -> str:
            return "run-42"

        result = asyncio.run(retry_async(func=returns_value, description="test_return"))

        assert result.ok is True
        assert result.value == "run-42"

    def test_exponential_backoff_between_retries(self) -> None:
        sleep_durations: list[float] = []

        async def record_sleep(seconds: float) -> None:
            sleep_durations.append(seconds)

        async def always_fail() -> None:
            raise RuntimeError("fail")

        asyncio.run(retry_async(
            func=always_fail,
            description="test_backoff",
            max_retries=4,
            sleep_fn=record_sleep,
        ))

        assert sleep_durations == [1.0, 2.0, 4.0]

    def test_no_sleep_on_immediate_success(self) -> None:
        sleep_durations: list[float] = []

        async def record_sleep(seconds: float) -> None:
            sleep_durations.append(seconds)

        async def succeed() -> str:
            return "ok"

        asyncio.run(retry_async(
            func=succeed,
            description="test_no_sleep",
            sleep_fn=record_sleep,
        ))

        assert sleep_durations == []

    def test_backoff_caps_at_max(self) -> None:
        sleep_durations: list[float] = []

        async def record_sleep(seconds: float) -> None:
            sleep_durations.append(seconds)

        async def always_fail() -> None:
            raise RuntimeError("fail")

        asyncio.run(retry_async(
            func=always_fail,
            description="test_cap",
            max_retries=8,
            sleep_fn=record_sleep,
        ))

        assert all(d <= 30.0 for d in sleep_durations)
        assert sleep_durations[-1] == 30.0


class TestStopClearSubmit:
    @pytest.mark.anyio
    async def test_happy_path_returns_true(self) -> None:
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()
        mini_wandb.log_step(run_id="r", step=1, metrics={"loss": 1.0})

        result = await stop_clear_submit(training_job, mini_wandb)

        assert result is True
        assert training_job._stopped
        assert training_job._submitted
        assert mini_wandb.latest(metric_name="loss") is None

    @pytest.mark.anyio
    async def test_stop_failure_but_job_stopped_still_submits(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.STOPPED])

        async def failing_stop(timeout_seconds: int = 300) -> None:
            raise RuntimeError("stop failed")

        training_job.stop_training = failing_stop  # type: ignore[assignment]

        result = await stop_clear_submit(training_job, MiniWandb())

        assert result is True
        assert training_job._submitted

    @pytest.mark.anyio
    async def test_stop_failure_job_still_running_skips_submit(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])

        async def failing_stop(timeout_seconds: int = 300) -> None:
            raise RuntimeError("stop failed")

        training_job.stop_training = failing_stop  # type: ignore[assignment]

        result = await stop_clear_submit(training_job, MiniWandb())

        assert result is False
        assert not training_job._submitted

    @pytest.mark.anyio
    async def test_submit_failure_returns_false(self) -> None:
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()

        async def failing_submit(excluded_node_ids: list[str] | None = None) -> str:
            raise RuntimeError("submit failed")

        training_job.submit_training = failing_submit  # type: ignore[assignment]

        result = await stop_clear_submit(training_job, mini_wandb)

        assert result is False
        assert training_job._stopped

    @pytest.mark.anyio
    async def test_excluded_node_ids_passed_to_submit(self) -> None:
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()

        result = await stop_clear_submit(
            training_job, mini_wandb, excluded_node_ids=["node-x", "node-y"],
        )

        assert result is True
        assert training_job._last_excluded_node_ids == ["node-x", "node-y"]

    @pytest.mark.anyio
    async def test_clear_happens_after_successful_submit(self) -> None:
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()
        mini_wandb.log_step(run_id="r", step=1, metrics={"loss": 2.0})

        result = await stop_clear_submit(training_job, mini_wandb)

        assert result is True
        assert mini_wandb.latest(metric_name="loss") is None

    @pytest.mark.anyio
    async def test_clear_not_called_on_submit_failure(self) -> None:
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()
        mini_wandb.log_step(run_id="r", step=1, metrics={"loss": 3.0})

        async def failing_submit(excluded_node_ids: list[str] | None = None) -> str:
            raise RuntimeError("submit failed")

        training_job.submit_training = failing_submit  # type: ignore[assignment]

        result = await stop_clear_submit(training_job, mini_wandb)

        assert result is False
        assert mini_wandb.latest(metric_name="loss") == 3.0

    @pytest.mark.anyio
    async def test_stop_failure_job_failed_still_submits(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.FAILED])

        async def failing_stop(timeout_seconds: int = 300) -> None:
            raise RuntimeError("stop failed")

        training_job.stop_training = failing_stop  # type: ignore[assignment]

        result = await stop_clear_submit(training_job, MiniWandb())

        assert result is True
        assert training_job._submitted
