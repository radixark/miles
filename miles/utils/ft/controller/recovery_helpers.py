"""Shared recovery primitives used by both FtController and RecoveryOrchestrator."""
from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.platform.protocols import NotificationProtocol, TrainingJobProtocol

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES: int = 3

_T = TypeVar("_T")


class _Sentinel:
    """Internal sentinel indicating retry_async exhausted all attempts."""


_EXHAUSTED = _Sentinel()


async def retry_async(
    func: Callable[[], Coroutine[Any, Any, _T]],
    description: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> _T | _Sentinel:
    """Retry an async callable up to *max_retries* times.

    Returns the callable's return value on success, or a sentinel object
    (falsy check via ``isinstance``) if all attempts are exhausted.
    Use :func:`retry_succeeded` to check the outcome.
    """
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception:
            logger.warning(
                "retry_failed description=%s attempt=%d/%d",
                description, attempt + 1, max_retries,
                exc_info=True,
            )
    logger.error("retry_exhausted description=%s", description)
    return _EXHAUSTED


def retry_succeeded(result: object) -> bool:
    """Return True if the retry_async call succeeded (did not exhaust retries)."""
    return not isinstance(result, _Sentinel)


async def stop_clear_submit(
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
) -> bool:
    """Stop training, clear metrics, submit new job. Returns True on success."""
    try:
        await training_job.stop_training()
    except Exception:
        logger.warning("stop_training_failed", exc_info=True)

    mini_wandb.clear()

    result = await retry_async(
        training_job.submit_training,
        description="submit_training",
    )
    return retry_succeeded(result)


async def safe_notify(
    notifier: NotificationProtocol | None,
    title: str,
    content: str,
    severity: str = "critical",
) -> None:
    if notifier is None:
        return
    try:
        await notifier.send(title=title, content=content, severity=severity)
    except Exception:
        logger.exception("notifier_send_failed title=%s", title)
