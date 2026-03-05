from __future__ import annotations

import asyncio

from miles.utils.ft.controller.recovery_helpers import retry_async


class TestRetryAsyncEdgePaths:
    def test_succeeds_on_second_attempt(self) -> None:
        call_count = 0

        async def flaky_fn() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")

        result = asyncio.run(retry_async(func=flaky_fn, description="test_retry"))
        assert result is True
        assert call_count == 2

    def test_all_retries_fail_returns_false(self) -> None:
        async def always_fail() -> None:
            raise RuntimeError("permanent error")

        result = asyncio.run(retry_async(
            func=always_fail, description="test_fail", max_retries=2,
        ))
        assert result is False
