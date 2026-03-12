"""Tests for state_machines/restart/utils.py (stop_and_submit) and utils/retry.py."""

from __future__ import annotations

import asyncio

import pytest
from tests.fast.utils.ft.conftest import FakeMainJob, make_failing_main_job

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.state_machines.restart.utils import stop_and_submit
from miles.utils.ft.controller.subsystem_hub import RestartMode
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
    async def test_on_main_job_new_run_called_after_successful_submit(self) -> None:
        main_job = FakeMainJob()
        calls: list[str] = []

        result = await stop_and_submit(
            main_job,
            on_main_job_new_run=lambda run_id: calls.append(run_id),
        )

        assert result is True
        assert len(calls) == 1
        assert calls[0].startswith("fake-")

    @pytest.mark.anyio
    async def test_on_main_job_new_run_not_called_on_submit_failure(self) -> None:
        main_job = make_failing_main_job(fail_submit=True)
        calls: list[str] = []

        result = await stop_and_submit(
            main_job,
            on_main_job_new_run=lambda run_id: calls.append(run_id),
        )

        assert result is False
        assert len(calls) == 0

    @pytest.mark.anyio
    async def test_on_main_job_new_run_not_called_for_subsystem_restart(self) -> None:
        main_job = FakeMainJob()
        calls: list[str] = []

        result = await stop_and_submit(
            main_job,
            on_main_job_new_run=lambda run_id: calls.append(run_id),
            restart_mode=RestartMode.SUBSYSTEM,
        )

        assert result is True
        assert len(calls) == 0

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
