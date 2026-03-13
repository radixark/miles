"""Tests for miles.utils.ft.utils.polling (poll_until)."""

from __future__ import annotations

import logging

import pytest

from miles.utils.ft.utils.polling import poll_until


# -----------------------------------------------------------------------
# Sync probe
# -----------------------------------------------------------------------


class TestPollUntilSyncProbe:
    @pytest.mark.anyio
    async def test_returns_immediately_when_predicate_satisfied(self) -> None:
        result = await poll_until(
            probe=lambda: 42,
            predicate=lambda v: v == 42,
            timeout=1.0,
            poll_interval=0.01,
            description="immediate",
        )
        assert result == 42

    @pytest.mark.anyio
    async def test_returns_after_several_polls(self) -> None:
        call_count = 0

        def counting_probe() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        result = await poll_until(
            probe=counting_probe,
            predicate=lambda v: v >= 3,
            timeout=2.0,
            poll_interval=0.01,
            description="after_several",
        )
        assert result >= 3
        assert call_count >= 3

    @pytest.mark.anyio
    async def test_raises_timeout_error_with_description(self) -> None:
        with pytest.raises(TimeoutError, match="never_true not met within"):
            await poll_until(
                probe=lambda: False,
                predicate=lambda v: v is True,
                timeout=0.05,
                poll_interval=0.01,
                description="never_true",
            )

    @pytest.mark.anyio
    async def test_returns_probe_value_on_success(self) -> None:
        """The actual probe return value (not just True) should be returned."""
        result = await poll_until(
            probe=lambda: {"status": "ready"},
            predicate=lambda v: v["status"] == "ready",
            timeout=1.0,
            poll_interval=0.01,
            description="dict_probe",
        )
        assert result == {"status": "ready"}


# -----------------------------------------------------------------------
# Async probe
# -----------------------------------------------------------------------


class TestPollUntilAsyncProbe:
    @pytest.mark.anyio
    async def test_async_probe_works(self) -> None:
        call_count = 0

        async def async_probe() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        result = await poll_until(
            probe=async_probe,
            predicate=lambda v: v >= 2,
            timeout=2.0,
            poll_interval=0.01,
            description="async_probe",
        )
        assert result >= 2

    @pytest.mark.anyio
    async def test_async_probe_timeout(self) -> None:
        async def always_zero() -> int:
            return 0

        with pytest.raises(TimeoutError, match="async_never"):
            await poll_until(
                probe=always_zero,
                predicate=lambda v: v > 0,
                timeout=0.05,
                poll_interval=0.01,
                description="async_never",
            )


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestPollUntilEdgeCases:
    @pytest.mark.anyio
    async def test_probe_exception_propagates(self) -> None:
        """Exceptions from probe should not be swallowed."""

        def exploding_probe() -> int:
            raise ConnectionError("network down")

        with pytest.raises(ConnectionError, match="network down"):
            await poll_until(
                probe=exploding_probe,
                predicate=lambda v: True,
                timeout=1.0,
                poll_interval=0.01,
                description="exploding",
            )

    @pytest.mark.anyio
    async def test_async_probe_exception_propagates(self) -> None:
        async def exploding_async_probe() -> int:
            raise ValueError("bad response")

        with pytest.raises(ValueError, match="bad response"):
            await poll_until(
                probe=exploding_async_probe,
                predicate=lambda v: True,
                timeout=1.0,
                poll_interval=0.01,
                description="async_exploding",
            )

    @pytest.mark.anyio
    async def test_zero_timeout_satisfied_returns_immediately(self) -> None:
        """With timeout=0, at least one probe runs. If predicate is satisfied, return."""
        probe_called = False

        def probe() -> bool:
            nonlocal probe_called
            probe_called = True
            return True

        result = await poll_until(
            probe=probe,
            predicate=lambda v: v is True,
            timeout=0,
            poll_interval=0.01,
            description="zero_timeout",
        )
        assert probe_called
        assert result is True

    @pytest.mark.anyio
    async def test_zero_timeout_not_satisfied_raises(self) -> None:
        """With timeout=0, probe runs once. If predicate fails, raise TimeoutError."""
        probe_called = False

        def probe() -> bool:
            nonlocal probe_called
            probe_called = True
            return False

        with pytest.raises(TimeoutError, match="zero_timeout"):
            await poll_until(
                probe=probe,
                predicate=lambda v: v is True,
                timeout=0,
                poll_interval=0.01,
                description="zero_timeout",
            )
        assert probe_called

    @pytest.mark.anyio
    async def test_remaining_time_less_than_poll_interval_still_probes(self) -> None:
        """Even when remaining time < poll_interval, one probe is guaranteed."""
        probe_count = 0

        def probe() -> int:
            nonlocal probe_count
            probe_count += 1
            return probe_count

        result = await poll_until(
            probe=probe,
            predicate=lambda v: v >= 1,
            timeout=0.001,
            poll_interval=10.0,
            description="short_budget",
        )
        assert result >= 1
        assert probe_count >= 1

    @pytest.mark.anyio
    async def test_negative_timeout_raises_value_error(self) -> None:
        """Negative timeout is rejected with ValueError (not silently treated as zero)."""
        with pytest.raises(ValueError, match="timeout must be >= 0"):
            await poll_until(
                probe=lambda: True,
                predicate=lambda v: v is True,
                timeout=-1.0,
                poll_interval=0.01,
                description="neg_timeout",
            )

    @pytest.mark.anyio
    async def test_zero_poll_interval_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="poll_interval must be > 0"):
            await poll_until(
                probe=lambda: True,
                predicate=lambda v: v is True,
                timeout=1.0,
                poll_interval=0,
                description="zero_interval",
            )

    @pytest.mark.anyio
    async def test_negative_poll_interval_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="poll_interval must be > 0"):
            await poll_until(
                probe=lambda: True,
                predicate=lambda v: v is True,
                timeout=1.0,
                poll_interval=-0.5,
                description="neg_interval",
            )

    @pytest.mark.anyio
    async def test_predicate_exception_propagates(self) -> None:
        def bad_predicate(v: int) -> bool:
            raise TypeError("predicate bug")

        with pytest.raises(TypeError, match="predicate bug"):
            await poll_until(
                probe=lambda: 1,
                predicate=bad_predicate,
                timeout=1.0,
                poll_interval=0.01,
                description="bad_pred",
            )


# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------


class TestPollUntilLogging:
    @pytest.mark.anyio
    async def test_logs_at_log_every_interval(self, caplog: pytest.LogCaptureFixture) -> None:
        call_count = 0

        def probe() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        with caplog.at_level(logging.INFO, logger="miles.utils.ft.utils.polling"):
            result = await poll_until(
                probe=probe,
                predicate=lambda v: v >= 8,
                timeout=2.0,
                poll_interval=0.001,
                description="log_test",
                log_every=3,
            )

        assert result >= 8
        log_messages = [r.message for r in caplog.records if "log_test" in r.message]
        assert len(log_messages) >= 1

    @pytest.mark.anyio
    async def test_no_logging_when_log_every_zero(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="miles.utils.ft.utils.polling"):
            await poll_until(
                probe=lambda: True,
                predicate=lambda v: v is True,
                timeout=1.0,
                poll_interval=0.01,
                description="no_log",
                log_every=0,
            )

        log_messages = [r.message for r in caplog.records if "no_log" in r.message]
        assert len(log_messages) == 0
