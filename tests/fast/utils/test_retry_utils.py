import asyncio
import logging

import pytest

from miles.utils import retry_utils
from miles.utils.retry_utils import NonRetryableError, retry, retry_until_deadline

pytestmark = pytest.mark.asyncio


class _FakeSleep:
    """Records sleep calls without actually sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class _FakeRandom:
    """Stands in for the random module and always returns the top of the requested range."""

    def __init__(self) -> None:
        self.ranges: list[tuple[float, float]] = []

    def uniform(self, low: float, high: float) -> float:
        self.ranges.append((low, high))
        return high


class TestRetryBasic:
    async def test_succeeds_immediately(self):
        """A successful first attempt does not sleep or retry."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1

        await retry(fn, sleep_fn=fake_sleep)

        assert call_count == 1
        assert fake_sleep.delays == []

    async def test_retries_then_succeeds(self):
        """Retry keeps attempting until the callback succeeds."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ValueError("not yet")

        await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep)

        assert call_count == 4
        assert len(fake_sleep.delays) == 3

    async def test_single_retry(self):
        """One failure followed by success produces one retry."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail once")

        await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep)

        assert call_count == 2
        assert len(fake_sleep.delays) == 1

    async def test_fn_receives_correct_attempt_number(self):
        """First call gets attempt=0, first retry gets attempt=1, etc."""
        received_attempts: list[int] = []
        fake_sleep = _FakeSleep()

        async def fn(attempt):
            received_attempts.append(attempt)
            if len(received_attempts) < 4:
                raise ValueError("not yet")

        await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep)

        assert received_attempts == [0, 1, 2, 3]


class TestRetryNonRetryable:
    async def test_non_retryable_error_is_raised_after_a_single_attempt(self):
        """A NonRetryableError aborts immediately: fn is called once and no backoff sleep happens."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt: int) -> None:
            nonlocal call_count
            call_count += 1
            raise NonRetryableError("cannot heal anymore")

        with pytest.raises(NonRetryableError, match="cannot heal anymore"):
            await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep, max_attempts=5)

        assert call_count == 1
        assert fake_sleep.delays == []

    async def test_ordinary_error_is_retried_up_to_max_attempts(self):
        """The same setup with an ordinary exception keeps retrying, proving the fast path is what stops it."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt: int) -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("transient")

        with pytest.raises(ValueError, match="transient"):
            await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep, max_attempts=5)

        assert call_count == 5
        assert len(fake_sleep.delays) == 4


class TestRetryLogging:
    async def test_logs_on_retry(self, caplog):
        """Each retry emits a warning log."""
        call_count = 0

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("boom")

        with caplog.at_level("WARNING"):
            await retry(fn, initial_delay=1.0, sleep_fn=_FakeSleep())

        retry_messages = [r for r in caplog.records if "retrying" in r.message]
        assert len(retry_messages) == 2

    async def test_no_log_on_first_success(self, caplog):
        """A successful first attempt emits no retry warning."""

        async def fn(_attempt):
            pass

        with caplog.at_level("WARNING"):
            await retry(fn, sleep_fn=_FakeSleep())

        assert not any("retrying" in r.message for r in caplog.records)

    async def test_logs_include_exc_info(self, caplog):
        """Retry warnings retain exception information."""
        call_count = 0

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("detail")

        with caplog.at_level("WARNING"):
            await retry(fn, initial_delay=1.0, sleep_fn=_FakeSleep())

        retry_records = [r for r in caplog.records if "retrying" in r.message]
        assert len(retry_records) == 1
        assert retry_records[0].exc_info is not None

    async def test_log_message_includes_delay(self, caplog):
        """A retry warning includes its sleep delay."""
        call_count = 0

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail")

        with caplog.at_level("WARNING"):
            await retry(fn, initial_delay=2.5, sleep_fn=_FakeSleep())

        retry_records = [r for r in caplog.records if "retrying" in r.message]
        assert len(retry_records) == 1
        assert "2.5s" in retry_records[0].message


class TestRetryMaxAttempts:
    async def test_raises_after_max_attempts(self):
        """The last exception propagates once max_attempts calls have all failed."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            raise ValueError(f"fail {call_count}")

        with pytest.raises(ValueError, match="fail 3"):
            await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep, max_attempts=3)

        assert call_count == 3
        assert len(fake_sleep.delays) == 2

    async def test_succeeds_on_last_allowed_attempt(self):
        """No exception when fn succeeds exactly at the max_attempts-th call."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("not yet")

        await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep, max_attempts=3)

        assert call_count == 3

    async def test_max_attempts_one_never_retries(self):
        """max_attempts=1 means a single call with no retry."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            await retry(fn, sleep_fn=fake_sleep, max_attempts=1)

        assert call_count == 1
        assert fake_sleep.delays == []

    async def test_default_is_unlimited(self):
        """Without max_attempts, retry keeps going far beyond any small cap."""
        call_count = 0

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            if call_count < 50:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=0.0, sleep_fn=_FakeSleep())

        assert call_count == 50

    async def test_invalid_max_attempts_rejected(self):
        """max_attempts below 1 is a programming error."""

        async def fn(_attempt):
            pass

        with pytest.raises(AssertionError):
            await retry(fn, sleep_fn=_FakeSleep(), max_attempts=0)

    async def test_gives_up_log_message(self, caplog):
        """The final failure logs a giving-up warning instead of a retrying one."""

        async def fn(_attempt):
            raise RuntimeError("fail")

        with caplog.at_level("WARNING"):
            with pytest.raises(RuntimeError):
                await retry(fn, initial_delay=1.0, sleep_fn=_FakeSleep(), max_attempts=2)

        assert any("giving up" in r.message for r in caplog.records)
        assert len([r for r in caplog.records if "retrying" in r.message]) == 1


class TestRetryBackoff:
    async def test_delay_doubles_each_retry(self):
        """Default exponential backoff doubles each delay."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=1.0, max_delay=100.0, backoff_factor=2.0, sleep_fn=fake_sleep)

        assert fake_sleep.delays == [1.0, 2.0, 4.0, 8.0]

    async def test_delay_capped_at_max(self):
        """Backoff delays stop growing at max_delay."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=1.0, max_delay=3.0, backoff_factor=2.0, sleep_fn=fake_sleep)

        assert fake_sleep.delays == [1.0, 2.0, 3.0, 3.0, 3.0]

    async def test_custom_backoff_factor(self):
        """A custom backoff factor scales each retry delay."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=1.0, max_delay=100.0, backoff_factor=3.0, sleep_fn=fake_sleep)

        assert fake_sleep.delays == [1.0, 3.0, 9.0]

    async def test_zero_initial_delay(self):
        """A zero initial delay keeps immediate retries nonblocking."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=0.0, sleep_fn=fake_sleep)

        assert call_count == 3
        assert fake_sleep.delays == [0.0, 0.0]

    async def test_default_params_are_reasonable(self):
        """The public retry defaults remain stable."""
        from miles.utils.retry_utils import _DEFAULT_BACKOFF_FACTOR, _DEFAULT_INITIAL_DELAY, _DEFAULT_MAX_DELAY

        assert _DEFAULT_INITIAL_DELAY == 1.0
        assert _DEFAULT_MAX_DELAY == 60.0
        assert _DEFAULT_BACKOFF_FACTOR == 2.0

    async def test_many_retries_stay_capped(self):
        """After hitting max_delay, all subsequent delays remain at max."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1
            if call_count <= 8:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=1.0, max_delay=5.0, backoff_factor=2.0, sleep_fn=fake_sleep)

        # 1, 2, 4, 5, 5, 5, 5, 5
        assert fake_sleep.delays == [1.0, 2.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0]


class TestRetryUntilDeadline:
    async def test_returns_the_value_of_the_first_success(self):
        """The helper hands back whatever the attempt returned."""
        result = await retry_until_deadline(lambda remaining: _immediately(7), total_seconds=1.0, retry_on=ValueError)
        assert result == 7

    async def test_retries_until_success(self):
        """A retryable failure is retried and the later success is returned."""
        attempts = []

        async def attempt(remaining: float) -> str:
            attempts.append(remaining)
            if len(attempts) < 3:
                raise ValueError("not yet")
            return "done"

        result = await retry_until_deadline(
            attempt, total_seconds=5.0, retry_on=ValueError, initial_delay=0.01, max_delay=0.05
        )
        assert result == "done" and len(attempts) == 3

    async def test_remaining_budget_shrinks(self):
        """Each attempt is told how much of the budget is left."""
        seen: list[float] = []

        async def attempt(remaining: float) -> None:
            seen.append(remaining)
            raise ValueError("always")

        with pytest.raises(ValueError):
            await retry_until_deadline(attempt, total_seconds=0.2, retry_on=ValueError, initial_delay=0.05)
        assert len(seen) >= 2 and seen[1] < seen[0]

    async def test_unlisted_error_propagates_immediately(self):
        """An error outside retry_on is not retried."""
        attempts = []

        async def attempt(remaining: float) -> None:
            attempts.append(1)
            raise TypeError("fatal")

        with pytest.raises(TypeError):
            await retry_until_deadline(attempt, total_seconds=5.0, retry_on=ValueError, initial_delay=0.01)
        assert len(attempts) == 1

    async def test_last_error_reraised_when_budget_runs_out(self):
        """The budget bounds the retries and the final failure surfaces."""

        async def attempt(remaining: float) -> None:
            raise ValueError("still down")

        with pytest.raises(ValueError, match="still down"):
            await retry_until_deadline(attempt, total_seconds=0.1, retry_on=ValueError, initial_delay=0.02)

    async def test_failed_attempts_emit_structured_info_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """Each failed attempt emits its structured retry diagnostics."""
        attempts = 0

        async def attempt(remaining: float) -> str:
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise ValueError(f"failure-{attempts}")
            return "done"

        with caplog.at_level(logging.INFO, logger="miles.utils.retry_utils"):
            result = await retry_until_deadline(
                attempt,
                total_seconds=1.0,
                retry_on=ValueError,
                initial_delay=0.0,
                jitter_ratio=0.0,
            )

        records = [
            record
            for record in caplog.records
            if "op=retry_until_deadline" in record.message and "phase=attempt_failed" in record.message
        ]
        assert result == "done"
        assert len(records) == 2
        for attempt_number, record in enumerate(records, start=1):
            assert f"attempt={attempt_number}" in record.message
            assert "sleep_s=0.0" in record.message
            assert "remaining_s=" in record.message
            assert f"error=ValueError('failure-{attempt_number}')" in record.message

    async def test_caller_log_fields_are_merged_into_the_attempt_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """Caller-supplied fields land in the retry log and may override the default op."""
        attempts = 0

        async def attempt(remaining: float) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ValueError("failure")
            return "done"

        with caplog.at_level(logging.INFO, logger="miles.utils.retry_utils"):
            await retry_until_deadline(
                attempt,
                total_seconds=1.0,
                retry_on=ValueError,
                initial_delay=0.0,
                jitter_ratio=0.0,
                log_fields={"op": "submit", "call": "c1"},
            )

        records = [record for record in caplog.records if "phase=attempt_failed" in record.message]
        assert len(records) == 1
        assert "op=submit" in records[0].message
        assert "call=c1" in records[0].message
        assert "op=retry_until_deadline" not in records[0].message

    async def test_backoff_grows_and_caps_before_each_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sleeps between retries grow by the backoff factor and then stop at max_delay."""
        fake_sleep = _FakeSleep()
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        attempts = 0

        async def attempt(remaining: float) -> str:
            nonlocal attempts
            attempts += 1
            if attempts <= 5:
                raise ValueError("still down")
            return "done"

        result = await retry_until_deadline(
            attempt,
            total_seconds=1000.0,
            retry_on=ValueError,
            initial_delay=0.5,
            max_delay=4.0,
            backoff_factor=3.0,
            jitter_ratio=0.0,
        )

        assert result == "done"
        assert fake_sleep.delays == [0.5, 1.5, 4.0, 4.0, 4.0]

    async def test_jitter_ratio_is_applied_to_each_base_delay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Jitter is drawn from zero to jitter_ratio times the base delay and added on top of it."""
        fake_sleep = _FakeSleep()
        fake_random = _FakeRandom()
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(retry_utils, "random", fake_random)
        attempts = 0

        async def attempt(remaining: float) -> str:
            nonlocal attempts
            attempts += 1
            if attempts <= 3:
                raise ValueError("still down")
            return "done"

        result = await retry_until_deadline(
            attempt,
            total_seconds=1000.0,
            retry_on=ValueError,
            initial_delay=1.0,
            max_delay=10.0,
            backoff_factor=2.0,
            jitter_ratio=0.25,
        )

        assert result == "done"
        assert fake_random.ranges == [(0.0, 0.25), (0.0, 0.5), (0.0, 1.0)]
        assert fake_sleep.delays == [1.25, 2.5, 5.0]

    async def test_budget_exhaustion_reraises_without_sleep_and_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A sleep that would outlast the remaining budget re-raises at once and logs the giving-up warning."""
        fake_sleep = _FakeSleep()
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        attempts = 0

        async def attempt(remaining: float) -> None:
            nonlocal attempts
            attempts += 1
            raise ValueError("still down")

        with caplog.at_level(logging.WARNING, logger="miles.utils.retry_utils"):
            with pytest.raises(ValueError, match="still down"):
                await retry_until_deadline(
                    attempt,
                    total_seconds=0.5,
                    retry_on=ValueError,
                    initial_delay=10.0,
                    jitter_ratio=0.0,
                )

        assert attempts == 1
        assert fake_sleep.delays == []
        records = [record for record in caplog.records if "giving up" in record.message]
        assert len(records) == 1
        assert records[0].message == "retry_until_deadline: giving up after 0.5s"
        assert records[0].exc_info is not None

    async def test_retry_on_tuple_retries_each_listed_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every member of a retry_on tuple, including OSError, is treated as retryable."""
        fake_sleep = _FakeSleep()
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        raised: list[type[Exception]] = []

        async def attempt(remaining: float) -> str:
            if not raised:
                raised.append(OSError)
                raise OSError("connection refused")
            if len(raised) == 1:
                raised.append(ValueError)
                raise ValueError("bad payload")
            return "done"

        result = await retry_until_deadline(
            attempt,
            total_seconds=1000.0,
            retry_on=(OSError, ValueError),
            initial_delay=0.5,
            jitter_ratio=0.0,
        )

        assert result == "done"
        assert raised == [OSError, ValueError]
        assert fake_sleep.delays == [0.5, 1.0]


async def _immediately(value):
    return value
