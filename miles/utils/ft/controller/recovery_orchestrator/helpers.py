"""Shared recovery primitives used by both FtController and RecoveryOrchestrator."""
from __future__ import annotations

import logging
from collections.abc import Callable

from miles.utils.ft.protocols.platform import JobStatus, NodeManagerProtocol, NotificationProtocol, TrainingJobProtocol
from miles.utils.ft.utils.retry import RetryResult, retry_async

logger = logging.getLogger(__name__)


async def stop_and_submit(
    training_job: TrainingJobProtocol,
    excluded_node_ids: list[str] | None = None,
    on_new_run: Callable[[str], None] | None = None,
) -> bool:
    """Stop training, submit new job, notify caller of new run_id. Returns True on success."""
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
    if result.ok and result.value is not None and on_new_run is not None:
        on_new_run(result.value)

    return result.ok


async def get_already_bad_nodes(node_manager: NodeManagerProtocol) -> set[str]:
    try:
        return set(await node_manager.get_bad_nodes())
    except Exception:
        logger.warning("get_bad_nodes_failed, proceeding without filter", exc_info=True)
        return set()


async def retry_mark_node_bad(
    node_manager: NodeManagerProtocol,
    node_id: str,
    reason: str,
) -> RetryResult[None]:
    return await retry_async(
        lambda: node_manager.mark_node_bad(node_id, reason=reason),
        description=f"mark_node_bad({node_id})",
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
