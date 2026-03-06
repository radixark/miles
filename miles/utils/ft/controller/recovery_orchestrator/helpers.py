"""Shared recovery primitives used by both FtController and RecoveryOrchestrator."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.protocols.platform import JobStatus, NotificationProtocol, TrainingJobProtocol

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES: int = 3
_MAX_BACKOFF_SECONDS: float = 30.0

_T = TypeVar("_T")


@dataclass(frozen=True)
class RetryResult(Generic[_T]):
    """Result of :func:`retry_async` — explicitly typed success/failure."""

    ok: bool
    value: _T | None = None
    error: str | None = None


async def retry_async(
    func: Callable[[], Coroutine[Any, Any, _T]],
    description: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep_fn: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
    per_call_timeout: float | None = None,
) -> RetryResult[_T]:
    """Retry an async callable up to *max_retries* times with exponential backoff.

    Returns a :class:`RetryResult` with ``ok=True`` and the return value on
    success, or ``ok=False`` with an error description if all attempts fail.

    When *per_call_timeout* is set, each individual call is wrapped with
    ``asyncio.wait_for`` to prevent a single hung invocation from blocking
    the entire retry loop.
    """
    last_error: str = ""

    for attempt in range(max_retries):
        try:
            coro = func()
            if per_call_timeout is not None:
                value = await asyncio.wait_for(coro, timeout=per_call_timeout)
            else:
                value = await coro
            return RetryResult(ok=True, value=value)
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "retry_failed description=%s attempt=%d/%d",
                description, attempt + 1, max_retries,
                exc_info=True,
            )
            if attempt < max_retries - 1:
                delay = min(2 ** attempt, _MAX_BACKOFF_SECONDS)
                await sleep_fn(delay)

    logger.error("retry_exhausted description=%s", description)
    return RetryResult(ok=False, error=f"exhausted {max_retries} retries: {last_error}")


async def stop_clear_submit(
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
    excluded_node_ids: list[str] | None = None,
) -> bool:
    """Stop training, clear metrics, submit new job. Returns True on success."""
    stop_result = await retry_async(
        training_job.stop_training,
        description="stop_training",
        max_retries=2,
    )

    if not stop_result.ok:
        try:
            status = await training_job.get_training_status()
        except Exception:
            logger.error("get_status_after_stop_failure_also_failed", exc_info=True)
            return False

        if status not in (JobStatus.STOPPED, JobStatus.FAILED):
            logger.error(
                "stop_training_failed_job_still_active status=%s, skipping submit",
                status.value,
            )
            return False

    result = await retry_async(
        lambda: training_job.submit_training(excluded_node_ids=excluded_node_ids),
        description="submit_training",
    )
    if result.ok:
        mini_wandb.clear()

    return result.ok


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
