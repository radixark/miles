import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_INITIAL_DELAY = 1.0
_DEFAULT_MAX_DELAY = 60.0
_DEFAULT_BACKOFF_FACTOR = 2.0


async def retry(
    fn: Callable[[int], Awaitable[Any]],
    *,
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    max_attempts: int | None = None,
) -> None:
    """Retry until ``fn`` does not throw, with exponential backoff."""
    assert max_attempts is None or max_attempts >= 1

    attempt = 0
    delay = initial_delay
    while True:
        try:
            await fn(attempt)
            return
        except Exception:
            attempt += 1
            if max_attempts is not None and attempt >= max_attempts:
                logger.warning(f"retry: attempt {attempt} failed, giving up (max_attempts={max_attempts})")
                raise
            logger.warning(f"retry: attempt {attempt} failed, retrying in {delay:.1f}s", exc_info=True)
            await sleep_fn(delay)
            delay = min(delay * backoff_factor, max_delay)


def retry_sync(
    fn: Callable[[], Any],
    *,
    should_retry: Callable[[Exception], bool],
    what: str,
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    sleep_fn: Callable[[float], None] | None = None,
    max_attempts: int = 3,
) -> Any:
    """Run ``fn`` with bounded retries, gated by ``should_retry``.

    ``fn`` must be safe to re-invoke on a retried failure — wrap an idempotent
    operation (e.g. a ``ray.get`` on already-submitted object refs), not a
    non-idempotent side effect. Errors ``should_retry`` rejects propagate on
    the first attempt. ``BaseException``s that are not ``Exception``s
    (``KeyboardInterrupt``, ``SystemExit``) are never intercepted.
    """
    assert max_attempts >= 1

    attempt = 0
    delay = initial_delay
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if not should_retry(e) or attempt >= max_attempts:
                if should_retry(e):
                    logger.warning(
                        f"retry ({what}): attempt {attempt} failed, giving up (max_attempts={max_attempts})"
                    )
                raise
            logger.warning(f"retry ({what}): attempt {attempt} failed, retrying in {delay:.1f}s", exc_info=True)
            (sleep_fn or time.sleep)(delay)
            delay = min(delay * backoff_factor, max_delay)
