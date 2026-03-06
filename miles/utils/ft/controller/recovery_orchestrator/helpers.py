"""Shared recovery primitives used by both FtController and RecoveryOrchestrator."""
from __future__ import annotations

import logging

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.protocols.platform import JobStatus, NotificationProtocol, TrainingJobProtocol
from miles.utils.ft.retry import retry_async

logger = logging.getLogger(__name__)


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
