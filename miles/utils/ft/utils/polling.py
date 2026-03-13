"""Async deadline-based polling primitive."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


async def poll_until(
    probe: Callable[[], _T | Awaitable[_T]],
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

    At least one probe is always executed, even when *timeout* is zero.
    This prevents a race where the operation has already completed but the
    remaining time budget is exhausted.

    Every *log_every* polls an info-level progress message is emitted.
    """
    if timeout < 0:
        raise ValueError(f"timeout must be >= 0, got {timeout}")
    if poll_interval <= 0:
        raise ValueError(f"poll_interval must be > 0, got {poll_interval}")

    logger.debug(
        "polling: poll_until starting description=%s, timeout=%s, poll_interval=%s",
        description, timeout, poll_interval,
    )
    deadline = time.monotonic() + timeout
    poll_count = 0

    while True:
        result = probe()
        if inspect.isawaitable(result):
            result = await result

        poll_count += 1
        if predicate(result):
            return result  # type: ignore[return-value]

        if time.monotonic() >= deadline:
            break

        if log_every > 0 and poll_count % log_every == 0:
            elapsed = timeout - (deadline - time.monotonic())
            logger.info(
                "poll_until description=%s elapsed=%.0fs poll_count=%d",
                description,
                elapsed,
                poll_count,
            )

        await asyncio.sleep(poll_interval)

    logger.warning("polling: poll_until timed out description=%s, timeout=%ss, poll_count=%d", description, timeout, poll_count)
    raise TimeoutError(f"{description} not met within {timeout}s")
