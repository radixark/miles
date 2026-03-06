"""Tests for miles.utils.ft.retry (retry_sync, retry_async, retry_async_or_raise)."""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest

from miles.utils.ft.retry import RetryResult, retry_async, retry_async_or_raise, retry_sync

_SLEEP_PATCH = "miles.utils.ft.retry.time.sleep"
_ASYNC_SLEEP_PATCH = "miles.utils.ft.retry.asyncio.sleep"


# -----------------------------------------------------------------------
# retry_sync
# -----------------------------------------------------------------------


class TestRetrySyncHappyPath:
    def test_immediate_success(self) -> None:
        result = retry_sync(func=lambda: "ok", description="test")
        assert result == RetryResult(ok=True, value="ok")

    def test_succeeds_on_second_attempt(self) -> None:
        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")
            return "recovered"

        with patch(_SLEEP_PATCH):
            result = retry_sync(func=flaky, description="flaky")

        assert result.ok is True
        assert result.value == "recovered"
        assert call_count == 2

    def test_no_sleep_on_immediate_success(self) -> None:
        sleep_calls: list[float] = []

        with patch(_SLEEP_PATCH, side_effect=lambda s: sleep_calls.append(s)):
            retry_sync(func=lambda: "fast", description="no_sleep")

        assert sleep_calls == []


class TestRetrySyncFailure:
    def test_all_retries_fail(self) -> None:
        with patch(_SLEEP_PATCH):
            result = retry_sync(
                func=lambda: (_ for _ in ()).throw(RuntimeError("permanent")),
                description="always_fail",
                max_retries=3,
            )

        assert result.ok is False
        assert result.error is not None
        assert "permanent" in result.error

    def test_returns_none_value_on_failure(self) -> None:
        with patch(_SLEEP_PATCH):
            result = retry_sync(
                func=lambda: (_ for _ in ()).throw(ValueError("err")),
                description="fail",
                max_retries=2,
            )

        assert result.value is None

    def test_max_retries_one_no_sleep(self) -> None:
        """With max_retries=1, should fail immediately without sleeping."""
        sleep_calls: list[float] = []

        with patch(_SLEEP_PATCH, side_effect=lambda s: sleep_calls.append(s)):
            result = retry_sync(
                func=lambda: (_ for _ in ()).throw(RuntimeError("once")),
                description="single",
                max_retries=1,
            )

        assert result.ok is False
        assert sleep_calls == []

    def test_error_message_includes_last_exception(self) -> None:
        call_count = 0

        def varying() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first")
            raise ValueError("second")

        with patch(_SLEEP_PATCH):
            result = retry_sync(func=varying, description="varying", max_retries=2)

        assert result.ok is False
        assert "second" in (result.error or "")


class TestRetrySyncBackoff:
    def test_fixed_delay_when_base_equals_max(self) -> None:
        sleep_calls: list[float] = []

        with patch(_SLEEP_PATCH, side_effect=lambda s: sleep_calls.append(s)):
            retry_sync(
                func=lambda: (_ for _ in ()).throw(RuntimeError("fail")),
                description="fixed",
                max_retries=4,
                backoff_base=0.5,
                max_backoff=0.5,
            )

        assert sleep_calls == [0.5, 0.5, 0.5]

    def test_exponential_backoff(self) -> None:
        sleep_calls: list[float] = []

        with patch(_SLEEP_PATCH, side_effect=lambda s: sleep_calls.append(s)):
            retry_sync(
                func=lambda: (_ for _ in ()).throw(RuntimeError("fail")),
                description="exponential",
                max_retries=4,
                backoff_base=1.0,
                max_backoff=30.0,
            )

        assert sleep_calls == [1.0, 2.0, 4.0]

    def test_backoff_caps_at_max(self) -> None:
        sleep_calls: list[float] = []

        with patch(_SLEEP_PATCH, side_effect=lambda s: sleep_calls.append(s)):
            retry_sync(
                func=lambda: (_ for _ in ()).throw(RuntimeError("fail")),
                description="cap",
                max_retries=8,
                backoff_base=1.0,
                max_backoff=10.0,
            )

        assert all(d <= 10.0 for d in sleep_calls)
        assert sleep_calls[-1] == 10.0


class TestRetrySyncLogging:
    def test_logs_warning_per_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        with patch(_SLEEP_PATCH), \
             caplog.at_level(logging.WARNING, logger="miles.utils.ft.retry"):
            retry_sync(
                func=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                description="log_check",
                max_retries=3,
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 3

    def test_logs_error_on_exhaustion(self, caplog: pytest.LogCaptureFixture) -> None:
        with patch(_SLEEP_PATCH), \
             caplog.at_level(logging.ERROR, logger="miles.utils.ft.retry"):
            retry_sync(
                func=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                description="exhaust_check",
                max_retries=2,
            )

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "retry_exhausted" in errors[0].message


# -----------------------------------------------------------------------
# retry_async
# -----------------------------------------------------------------------


class TestRetryAsyncPerCallTimeout:
    @pytest.mark.anyio
    async def test_per_call_timeout_aborts_hung_call(self) -> None:
        async def hang() -> str:
            await asyncio.sleep(100)
            return "never"

        result = await retry_async(
            func=hang, description="hung", max_retries=1, per_call_timeout=0.01,
        )
        assert result.ok is False
        assert result.error is not None

    @pytest.mark.anyio
    async def test_per_call_timeout_retries_after_timeout(self) -> None:
        call_count = 0

        async def slow_then_fast() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(100)
            return "ok"

        result = await retry_async(
            func=slow_then_fast, description="recover",
            max_retries=2, per_call_timeout=0.01,
        )
        assert result.ok is True
        assert result.value == "ok"
        assert call_count == 2


# -----------------------------------------------------------------------
# retry_async_or_raise
# -----------------------------------------------------------------------


class TestRetryAsyncOrRaiseHappyPath:
    @pytest.mark.anyio
    async def test_immediate_success_returns_value(self) -> None:
        async def fn() -> str:
            return "ok"

        result = await retry_async_or_raise(func=fn, description="test")
        assert result == "ok"

    @pytest.mark.anyio
    async def test_succeeds_on_third_attempt(self) -> None:
        call_count = 0

        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "recovered"

        result = await retry_async_or_raise(
            func=flaky, description="flaky", max_retries=3,
        )
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.anyio
    async def test_no_sleep_on_immediate_success(self) -> None:
        mock_sleep = AsyncMock()

        async def succeed() -> str:
            return "fast"

        with patch(_ASYNC_SLEEP_PATCH, mock_sleep):
            await retry_async_or_raise(func=succeed, description="no_sleep")

        mock_sleep.assert_not_called()


class TestRetryAsyncOrRaiseFailure:
    @pytest.mark.anyio
    async def test_raises_last_exception_on_exhaustion(self) -> None:
        async def always_fail() -> str:
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            await retry_async_or_raise(
                func=always_fail, description="fail", max_retries=2,
            )

    @pytest.mark.anyio
    async def test_raises_the_last_specific_exception(self) -> None:
        call_count = 0

        async def varying_error() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first")
            raise TypeError("second")

        with pytest.raises(TypeError, match="second"):
            await retry_async_or_raise(
                func=varying_error, description="varying", max_retries=2,
            )

    @pytest.mark.anyio
    async def test_max_retries_one_raises_immediately(self) -> None:
        mock_sleep = AsyncMock()

        async def fail() -> str:
            raise RuntimeError("once")

        with patch(_ASYNC_SLEEP_PATCH, mock_sleep), \
             pytest.raises(RuntimeError, match="once"):
            await retry_async_or_raise(func=fail, description="single", max_retries=1)

        mock_sleep.assert_not_called()


class TestRetryAsyncOrRaiseBackoff:
    @pytest.mark.anyio
    async def test_exponential_backoff_delays(self) -> None:
        sleep_calls: list[float] = []
        mock_sleep = AsyncMock(side_effect=lambda s: sleep_calls.append(s))

        async def fail() -> str:
            raise RuntimeError("fail")

        with patch(_ASYNC_SLEEP_PATCH, mock_sleep), \
             pytest.raises(RuntimeError):
            await retry_async_or_raise(
                func=fail, description="backoff",
                max_retries=4, backoff_base=1.0, max_backoff=30.0,
            )

        assert sleep_calls == [1.0, 2.0, 4.0]

    @pytest.mark.anyio
    async def test_backoff_caps_at_max(self) -> None:
        sleep_calls: list[float] = []
        mock_sleep = AsyncMock(side_effect=lambda s: sleep_calls.append(s))

        async def fail() -> str:
            raise RuntimeError("fail")

        with patch(_ASYNC_SLEEP_PATCH, mock_sleep), \
             pytest.raises(RuntimeError):
            await retry_async_or_raise(
                func=fail, description="cap",
                max_retries=6, backoff_base=2.0, max_backoff=10.0,
            )

        assert all(d <= 10.0 for d in sleep_calls)
        assert sleep_calls[-1] == 10.0


class TestRetryAsyncOrRaisePerCallTimeout:
    @pytest.mark.anyio
    async def test_per_call_timeout_triggers(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(10)
            return "late"

        with pytest.raises(asyncio.TimeoutError):
            await retry_async_or_raise(
                func=slow,
                description="slow",
                max_retries=1,
                per_call_timeout=0.01,
            )

    @pytest.mark.anyio
    async def test_per_call_timeout_retries_then_succeeds(self) -> None:
        call_count = 0

        async def slow_then_fast() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(100)
            return "ok"

        result = await retry_async_or_raise(
            func=slow_then_fast, description="timeout_recover",
            max_retries=2, per_call_timeout=0.01,
        )
        assert result == "ok"
        assert call_count == 2
