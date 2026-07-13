import asyncio
import logging
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
