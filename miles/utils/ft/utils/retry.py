"""Retry primitives for both async and sync callables."""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES: int = 3
_MAX_BACKOFF_SECONDS: float = 30.0

_T = TypeVar("_T")


@dataclass(frozen=True)
class RetryResult(Generic[_T]):
    """Result of :func:`retry_async` — explicitly typed success/failure."""

    ok: bool
    value: _T | None = None
    exception: Exception | None = None


async def retry_async(
    func: Callable[[], Coroutine[Any, Any, _T]],
    description: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep_fn: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
    per_call_timeout: float | None = None,
    backoff_base: float = 1.0,
    max_backoff: float = _MAX_BACKOFF_SECONDS,
) -> RetryResult[_T]:
    """Retry an async callable up to *max_retries* times with exponential backoff.

    Returns a :class:`RetryResult` with ``ok=True`` and the return value on
    success, or ``ok=False`` with the exception if all attempts fail.

    Backoff formula: ``min(backoff_base * 2**attempt, max_backoff)``.

    When *per_call_timeout* is set, each individual call is wrapped with
    ``asyncio.wait_for`` to prevent a single hung invocation from blocking
    the entire retry loop.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            coro = func()
            if per_call_timeout is not None:
                value = await asyncio.wait_for(coro, timeout=per_call_timeout)
            else:
                value = await coro
            return RetryResult(ok=True, value=value)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "retry_failed description=%s attempt=%d/%d",
                description, attempt + 1, max_retries,
                exc_info=True,
            )
            if attempt < max_retries - 1:
                delay = min(backoff_base * (2 ** attempt), max_backoff)
                await sleep_fn(delay)

    logger.error("retry_exhausted description=%s", description)
    return RetryResult(ok=False, exception=last_exc)


async def retry_async_or_raise(
    func: Callable[[], Coroutine[Any, Any, _T]],
    description: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    per_call_timeout: float | None = None,
    backoff_base: float = 1.0,
    max_backoff: float = _MAX_BACKOFF_SECONDS,
) -> _T:
    """Like :func:`retry_async` but raises the last exception on exhaustion.

    Thin wrapper around :func:`retry_async` for callers that need the
    original exception rather than a :class:`RetryResult`.
    """
    result = await retry_async(
        func=func,
        description=description,
        max_retries=max_retries,
        per_call_timeout=per_call_timeout,
        backoff_base=backoff_base,
        max_backoff=max_backoff,
    )
    if result.ok:
        return result.value  # type: ignore[return-value]
    raise result.exception  # type: ignore[misc]


def retry_sync(
    func: Callable[[], _T],
    description: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = 1.0,
    max_backoff: float = _MAX_BACKOFF_SECONDS,
) -> RetryResult[_T]:
    """Synchronous retry with configurable backoff.

    Backoff formula: ``min(backoff_base * 2**attempt, max_backoff)``.
    For fixed delay set ``backoff_base == max_backoff``.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            value = func()
            return RetryResult(ok=True, value=value)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "retry_failed description=%s attempt=%d/%d",
                description, attempt + 1, max_retries,
                exc_info=True,
            )
            if attempt < max_retries - 1:
                delay = min(backoff_base * (2 ** attempt), max_backoff)
                time.sleep(delay)

    logger.error("retry_exhausted description=%s", description)
    return RetryResult(ok=False, exception=last_exc)
