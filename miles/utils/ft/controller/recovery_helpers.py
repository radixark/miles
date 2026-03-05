"""Shared recovery primitives used by both FtController and RecoveryOrchestrator."""
from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.platform.protocols import NotificationProtocol, TrainingJobProtocol

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES: int = 3


async def retry_async(
    func: Callable[[], Coroutine[Any, Any, Any]],
    description: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> bool:
    for attempt in range(max_retries):
        try:
            await func()
            return True
        except Exception:
            logger.warning(
                "retry_failed description=%s attempt=%d/%d",
                description, attempt + 1, max_retries,
                exc_info=True,
            )
    logger.error("retry_exhausted description=%s", description)
    return False


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

    return await retry_async(
        training_job.submit_training,
        description="submit_training",
    )


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
