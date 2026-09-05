import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)

_DEFAULT_INITIAL_DELAY = 1.0
_DEFAULT_MAX_DELAY = 60.0
_DEFAULT_BACKOFF_FACTOR = 2.0
_DEFAULT_JITTER_RATIO = 0.1

_T = TypeVar("_T")


class NonRetryableError(Exception):
    pass


class AttemptTimeoutError(TimeoutError):
    pass


async def retry(
    fn: Callable[[int], Awaitable[_T]],
    *,
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    max_attempts: int | None = None,
) -> _T:
    """Retry until ``fn`` does not throw, with exponential backoff."""
    assert max_attempts is None or max_attempts >= 1

    attempt = 0
    delay = initial_delay
    while True:
        try:
            return await fn(attempt)
        except NonRetryableError:
            raise
        except Exception:
            attempt += 1
            if max_attempts is not None and attempt >= max_attempts:
                logger.warning(f"retry: attempt {attempt} failed, giving up (max_attempts={max_attempts})")
                raise
            logger.warning(f"retry: attempt {attempt} failed, retrying in {delay:.1f}s", exc_info=True)
            await sleep_fn(delay)
            delay = min(delay * backoff_factor, max_delay)


async def retry_until_deadline(
    fn: Callable[[float], Awaitable[_T]],
    *,
    total_seconds: float,
    retry_on: type[Exception] | tuple[type[Exception], ...],
    attempt_seconds: float | None = None,
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    jitter_ratio: float = _DEFAULT_JITTER_RATIO,
    log_fields: dict[str, Any] | None = None,
) -> _T:
    expires_at = time.monotonic() + total_seconds
    retryable = retry_on if isinstance(retry_on, tuple) else (retry_on,)
    delay = initial_delay
    attempt = 0

    while True:
        try:
            return await _run_attempt(
                fn, remaining=max(0.0, expires_at - time.monotonic()), attempt_seconds=attempt_seconds
            )
        except (AttemptTimeoutError, *retryable) as e:
            attempt += 1
            sleep_seconds = delay + random.uniform(0.0, delay * jitter_ratio)
            if expires_at - time.monotonic() <= sleep_seconds:
                logger.warning(f"retry_until_deadline: giving up after {total_seconds:.1f}s", exc_info=True)
                raise
            log_structured(
                logger.info,
                **{"op": "retry_until_deadline", **(log_fields or {})},
                phase="retrying",
                attempt=attempt,
                sleep_s=round(sleep_seconds, 3),
                remaining_s=round(expires_at - time.monotonic(), 3),
                error=repr(e),
            )
            await asyncio.sleep(sleep_seconds)
            delay = min(delay * backoff_factor, max_delay)


async def _run_attempt(fn: Callable[[float], Awaitable[_T]], *, remaining: float, attempt_seconds: float | None) -> _T:
    if attempt_seconds is None:
        return await fn(remaining)

    budget = min(attempt_seconds, remaining)
    attempt = asyncio.ensure_future(fn(budget))
    try:
        return await asyncio.wait_for(attempt, timeout=budget)
    except (TimeoutError, asyncio.TimeoutError) as e:
        if not attempt.cancelled():
            raise
        raise AttemptTimeoutError(f"one attempt did not answer within {budget:.2f}s") from e
