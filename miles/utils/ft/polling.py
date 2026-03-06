"""Async deadline-based polling primitive."""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar, Union

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


async def poll_until(
    probe: Callable[[], Union[_T, Awaitable[_T]]],
    predicate: Callable[[_T], bool],
    *,
    timeout: float,
    poll_interval: float = 5.0,
    description: str = "poll_until",
    log_every: int = 6,
) -> _T:
    """Poll *probe* until *predicate* returns True, or raise :class:`TimeoutError`.

    *probe* may be sync or async — if the return value is awaitable it is
    automatically awaited.

    Every *log_every* polls an info-level progress message is emitted.
    """
    deadline = time.monotonic() + timeout
    poll_count = 0

    while time.monotonic() < deadline:
        result = probe()
        if inspect.isawaitable(result):
            result = await result

        poll_count += 1
        if predicate(result):
            return result  # type: ignore[return-value]

        if log_every > 0 and poll_count % log_every == 0:
            elapsed = timeout - (deadline - time.monotonic())
            logger.info(
                "poll_until description=%s elapsed=%.0fs poll_count=%d",
                description, elapsed, poll_count,
            )

        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"{description} not met within {timeout}s")
