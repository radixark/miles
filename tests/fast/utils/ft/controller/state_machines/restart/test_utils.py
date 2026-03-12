"""Tests for state_machines/restart/utils.py (stop_and_submit) and utils/retry.py."""

from __future__ import annotations

import asyncio

import pytest
from tests.fast.utils.ft.conftest import FakeMainJob, make_failing_main_job

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.state_machines.restart.utils import stop_and_submit
from miles.utils.ft.utils.retry import RetryResult, retry_async


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

        result = asyncio.run(
            retry_async(
                func=always_fail,
                description="test_fail",
                max_retries=2,
            )
        )

        assert isinstance(result, RetryResult)
        assert result.ok is False
        assert result.exception is not None
        assert "permanent error" in str(result.exception)

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

        asyncio.run(
            retry_async(
                func=always_fail,
                description="test_backoff",
                max_retries=4,
                sleep_fn=record_sleep,
            )
        )

        assert sleep_durations == [1.0, 2.0, 4.0]

    def test_no_sleep_on_immediate_success(self) -> None:
        sleep_durations: list[float] = []

        async def record_sleep(seconds: float) -> None:
            sleep_durations.append(seconds)

        async def succeed() -> str:
            return "ok"

        asyncio.run(
            retry_async(
                func=succeed,
                description="test_no_sleep",
                sleep_fn=record_sleep,
            )
        )

        assert sleep_durations == []

    def test_backoff_caps_at_max(self) -> None:
        sleep_durations: list[float] = []

        async def record_sleep(seconds: float) -> None:
            sleep_durations.append(seconds)

        async def always_fail() -> None:
            raise RuntimeError("fail")

        asyncio.run(
            retry_async(
                func=always_fail,
                description="test_cap",
                max_retries=8,
                sleep_fn=record_sleep,
            )
        )

        assert all(d <= 30.0 for d in sleep_durations)
        assert sleep_durations[-1] == 30.0


class TestStopAndSubmit:
    @pytest.mark.anyio
    async def test_happy_path_returns_true(self) -> None:
        main_job = FakeMainJob()

        result = await stop_and_submit(main_job)

        assert result is True
        assert main_job._stopped
        assert main_job._submitted

    @pytest.mark.anyio
    async def test_stop_failure_but_job_stopped_still_submits(self) -> None:
        main_job = make_failing_main_job(
            fail_stop=True,
            status_sequence=[JobStatus.STOPPED],
        )

        result = await stop_and_submit(main_job)

        assert result is True
        assert main_job._submitted

    @pytest.mark.anyio
    async def test_stop_failure_job_still_running_skips_submit(self) -> None:
        main_job = make_failing_main_job(
            fail_stop=True,
            status_sequence=[JobStatus.RUNNING],
        )

        result = await stop_and_submit(main_job)

        assert result is False
        assert not main_job._submitted

    @pytest.mark.anyio
    async def test_submit_failure_returns_false(self) -> None:
        main_job = make_failing_main_job(fail_submit=True)

        result = await stop_and_submit(main_job)

        assert result is False
        assert main_job._stopped

    @pytest.mark.anyio
    async def test_on_new_run_called_after_successful_submit(self) -> None:
        """on_new_run callback receives the run_id on success."""
        main_job = FakeMainJob()
        calls: list[str] = []

        result = await stop_and_submit(
            main_job,
            on_new_run=lambda run_id: calls.append(run_id),
        )

        assert result is True
        assert len(calls) == 1
        assert calls[0].startswith("fake-")

    @pytest.mark.anyio
    async def test_on_new_run_not_called_on_submit_failure(self) -> None:
        main_job = make_failing_main_job(fail_submit=True)
        calls: list[str] = []

        result = await stop_and_submit(
            main_job,
            on_new_run=lambda run_id: calls.append(run_id),
        )

        assert result is False
        assert len(calls) == 0

    @pytest.mark.anyio
    async def test_on_new_run_omitted_for_subsystem_restart(self) -> None:
        """Subsystem restarts omit the on_new_run callback entirely,
        so the callback is never invoked regardless of restart_mode.
        Previously stop_and_submit checked restart_mode internally;
        now callers simply don't pass on_new_run for subsystems."""
        main_job = FakeMainJob()

        result = await stop_and_submit(main_job, on_new_run=None)

        assert result is True

    @pytest.mark.anyio
    async def test_stop_failure_job_failed_still_submits(self) -> None:
        main_job = make_failing_main_job(
            fail_stop=True,
            status_sequence=[JobStatus.FAILED],
        )

        result = await stop_and_submit(main_job)

        assert result is True
        assert main_job._submitted

    @pytest.mark.anyio
    async def test_submit_called_exactly_once(self) -> None:
        """stop_and_submit calls submit_job exactly once (no outer retry)."""
        main_job = FakeMainJob()

        result = await stop_and_submit(main_job)

        assert result is True
        assert main_job._submit_call_count == 1

    @pytest.mark.anyio
    async def test_submit_exception_returns_false_without_retry(self) -> None:
        """When submit_job raises, stop_and_submit returns False without retrying."""
        main_job = make_failing_main_job(fail_submit=True)

        result = await stop_and_submit(main_job)

        assert result is False


class TestRestartLockSerialization:
    """restart_lock serializes concurrent stop_and_submit calls.

    Without the lock, two concurrent callers could interleave their
    stop->start sequences: both stop, then both start, resulting in
    a ghost job that the controller no longer tracks.
    """

    @pytest.mark.anyio
    async def test_concurrent_calls_are_serialized(self) -> None:
        """Two concurrent stop_and_submit with the same lock run sequentially."""
        lock = asyncio.Lock()
        execution_log: list[str] = []

        class SlowJob(FakeMainJob):
            async def stop(self, timeout_seconds: int = 300) -> None:
                execution_log.append("stop_start")
                await asyncio.sleep(0.05)
                execution_log.append("stop_end")
                await super().stop(timeout_seconds=timeout_seconds)

            async def start(self) -> str:
                execution_log.append("start_start")
                run_id = await super().start()
                execution_log.append("start_end")
                return run_id

        job = SlowJob()

        task1 = asyncio.create_task(stop_and_submit(job, restart_lock=lock))
        task2 = asyncio.create_task(stop_and_submit(job, restart_lock=lock))
        results = await asyncio.gather(task1, task2)

        # One must succeed, the other may fail (double submit guard) or succeed
        assert any(r is True for r in results)

        # The stop->start pairs must not interleave.
        # With proper serialization: stop_start, stop_end, start_start, start_end,
        # then second caller's sequence follows entirely after.
        stop_starts = [i for i, e in enumerate(execution_log) if e == "stop_start"]
        start_ends = [i for i, e in enumerate(execution_log) if e == "start_end"]
        assert len(stop_starts) >= 1
        assert len(start_ends) >= 1
        # First caller's start_end must come before second caller's stop_start
        if len(stop_starts) >= 2:
            assert start_ends[0] < stop_starts[1]

    @pytest.mark.anyio
    async def test_no_lock_allows_normal_execution(self) -> None:
        """When restart_lock is None, stop_and_submit still works normally."""
        main_job = FakeMainJob()

        result = await stop_and_submit(main_job, restart_lock=None)

        assert result is True
