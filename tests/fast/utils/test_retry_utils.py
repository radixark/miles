import pytest

from miles.utils.retry_utils import retry

pytestmark = pytest.mark.asyncio


class _FakeSleep:
    """Records sleep calls without actually sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class TestRetryBasic:
    async def test_succeeds_immediately(self):
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            nonlocal call_count
            call_count += 1

        await retry(fn, sleep_fn=fake_sleep)

        assert call_count == 1
        assert fake_sleep.delays == []

    async def test_retries_then_succeeds(self):
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


class TestRetryLogging:
    async def test_logs_on_retry(self, caplog):
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
        async def fn(_attempt):
            pass

        with caplog.at_level("WARNING"):
            await retry(fn, sleep_fn=_FakeSleep())

        assert not any("retrying" in r.message for r in caplog.records)

    async def test_logs_include_exc_info(self, caplog):
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


class TestRetryBackoff:
    async def test_delay_doubles_each_retry(self):
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
