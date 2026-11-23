import logging

import pytest

from miles.utils.retry_utils import retry_until_deadline_sync


class TestSynchronousRetry:
    def test_transient_errors_retry_until_success_without_replacing_the_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retryable failures preserve the successful operation's exact result."""
        clock = iter([0.0, 0.0, 1.0])
        sleeps: list[float] = []
        attempts: list[int] = []
        result = object()
        monkeypatch.setattr("miles.utils.retry_utils.time.monotonic", lambda: next(clock))
        monkeypatch.setattr("miles.utils.retry_utils.time.sleep", sleeps.append)

        def operation() -> object:
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("temporary")
            return result

        actual = retry_until_deadline_sync(
            operation, total_seconds=5, retry_if=lambda error: isinstance(error, ConnectionError), interval_seconds=1
        )

        assert actual is result
        assert len(attempts) == 3
        assert sleeps == [1, 1]

    def test_sleep_is_clipped_to_the_remaining_budget_and_last_error_is_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A retry cannot sleep beyond its deadline or replace the final operation error."""
        clock = iter([0.0, 3.0, 5.0])
        sleeps: list[float] = []
        error = ConnectionError("last error")
        monkeypatch.setattr("miles.utils.retry_utils.time.monotonic", lambda: next(clock))
        monkeypatch.setattr("miles.utils.retry_utils.time.sleep", sleeps.append)

        def operation() -> None:
            raise error

        with pytest.raises(ConnectionError) as captured:
            retry_until_deadline_sync(operation, total_seconds=5, retry_if=lambda error: True, interval_seconds=10)

        assert captured.value is error
        assert sleeps == [2]

    def test_nonretryable_errors_escape_without_sleeping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The retry predicate prevents retries for permanent errors."""
        sleeps: list[float] = []
        monkeypatch.setattr("miles.utils.retry_utils.time.sleep", sleeps.append)
        error = ValueError("permanent")

        def operation() -> None:
            raise error

        with pytest.raises(ValueError) as captured:
            retry_until_deadline_sync(operation, total_seconds=100, retry_if=lambda error: False)

        assert captured.value is error
        assert sleeps == []

    def test_retry_warning_keeps_the_original_traceback(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A recovered transient failure remains diagnosable from its logged traceback."""
        results = iter([False, True])
        monkeypatch.setattr("miles.utils.retry_utils.time.sleep", lambda duration: None)

        def operation() -> int:
            if not next(results):
                raise ConnectionError("recoverable")
            return 7

        with caplog.at_level(logging.WARNING):
            assert retry_until_deadline_sync(operation, total_seconds=5, retry_if=lambda error: True) == 7

        [record] = [record for record in caplog.records if "Operation failed" in record.message]
        assert record.exc_info is not None
        assert record.exc_info[0] is ConnectionError
