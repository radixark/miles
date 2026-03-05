"""Shared recovery primitives used by both FtController and RecoveryOrchestrator."""
from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.platform.protocols import JobStatus, NotificationProtocol, TrainingJobProtocol

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES: int = 3

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
) -> RetryResult[_T]:
    """Retry an async callable up to *max_retries* times.

    Returns a :class:`RetryResult` with ``ok=True`` and the return value on
    success, or ``ok=False`` with an error description if all attempts fail.
    """
    last_error: str = ""

    for attempt in range(max_retries):
        try:
            value = await func()
            return RetryResult(ok=True, value=value)
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "retry_failed description=%s attempt=%d/%d",
                description, attempt + 1, max_retries,
                exc_info=True,
            )

    logger.error("retry_exhausted description=%s", description)
    return RetryResult(ok=False, error=f"exhausted {max_retries} retries: {last_error}")


async def stop_clear_submit(
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
    excluded_node_ids: list[str] | None = None,
) -> bool:
    """Stop training, clear metrics, submit new job. Returns True on success."""
    try:
        await training_job.stop_training()
    except Exception:
        logger.warning("stop_training_failed", exc_info=True)
        status = await training_job.get_training_status()
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
