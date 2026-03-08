"""Tests for the graceful_degrade decorator (sync, async, FaultInjectionError)."""

from __future__ import annotations

import logging

import pytest

from miles.utils.ft.utils.graceful_degrade import (
    FaultInjectionError,
    _format_call_args,
    graceful_degrade,
)


# ---------------------------------------------------------------------------
# _format_call_args
# ---------------------------------------------------------------------------


class TestFormatCallArgs:
    def test_simple_args(self) -> None:
        import inspect

        def fn(x: int, y: str) -> None: ...

        sig = inspect.signature(fn)
        result = _format_call_args(sig, (1, "hello"), {})
        assert "x=1" in result
        assert "y='hello'" in result

    def test_self_param_skipped(self) -> None:
        import inspect

        def fn(self: object, x: int) -> None: ...

        sig = inspect.signature(fn)
        result = _format_call_args(sig, (None, 42), {})
        assert "self" not in result
        assert "x=42" in result

    def test_long_repr_truncated(self) -> None:
        import inspect

        def fn(data: str) -> None: ...

        sig = inspect.signature(fn)
        long_str = "a" * 500
        result = _format_call_args(sig, (long_str,), {})
        assert "..." in result
        assert len(result) < 500

    def test_returns_empty_on_no_params(self) -> None:
        import inspect

        def fn() -> None: ...

        sig = inspect.signature(fn)
        result = _format_call_args(sig, (), {})
        assert result == ""


# ---------------------------------------------------------------------------
# Sync decorator
# ---------------------------------------------------------------------------


class TestGracefulDegradeSync:
    def test_success_passes_through(self) -> None:
        @graceful_degrade(default=-1)
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_exception_returns_default(self) -> None:
        @graceful_degrade(default=-1)
        def boom() -> int:
            raise ValueError("oops")

        assert boom() == -1

    def test_default_is_none_when_omitted(self) -> None:
        @graceful_degrade()
        def boom() -> int:
            raise ValueError("oops")

        assert boom() is None

    def test_fault_injection_error_not_caught(self) -> None:
        @graceful_degrade(default=-1)
        def injected() -> int:
            raise FaultInjectionError("test injection")

        with pytest.raises(FaultInjectionError, match="test injection"):
            injected()

    def test_custom_log_message(self, caplog: pytest.LogCaptureFixture) -> None:
        @graceful_degrade(default=0, msg="custom_fail_msg")
        def boom() -> int:
            raise RuntimeError("bang")

        with caplog.at_level(logging.WARNING):
            result = boom()

        assert result == 0
        assert "custom_fail_msg" in caplog.text

    def test_custom_log_level(self, caplog: pytest.LogCaptureFixture) -> None:
        @graceful_degrade(default=0, log_level=logging.ERROR)
        def boom() -> int:
            raise RuntimeError("bang")

        with caplog.at_level(logging.ERROR):
            boom()

        assert any(r.levelno == logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# Async decorator
# ---------------------------------------------------------------------------


class TestGracefulDegradeAsync:
    @pytest.mark.asyncio
    async def test_success_passes_through(self) -> None:
        @graceful_degrade(default=-1)
        async def add(a: int, b: int) -> int:
            return a + b

        assert await add(2, 3) == 5

    @pytest.mark.asyncio
    async def test_exception_returns_default(self) -> None:
        @graceful_degrade(default=-1)
        async def boom() -> int:
            raise ValueError("oops")

        assert await boom() == -1

    @pytest.mark.asyncio
    async def test_fault_injection_error_not_caught(self) -> None:
        @graceful_degrade(default=-1)
        async def injected() -> int:
            raise FaultInjectionError("async injection")

        with pytest.raises(FaultInjectionError, match="async injection"):
            await injected()

    @pytest.mark.asyncio
    async def test_preserves_function_name(self) -> None:
        @graceful_degrade(default=None)
        async def my_original_func() -> None: ...

        assert my_original_func.__name__ == "my_original_func"
