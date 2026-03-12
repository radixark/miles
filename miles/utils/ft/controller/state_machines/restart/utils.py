from __future__ import annotations

import logging
from collections.abc import Callable

from miles.utils.ft.adapters.types import JobStatus, NodeManagerProtocol, StoppableJobProtocol
from miles.utils.ft.utils.retry import RetryResult, retry_async

logger = logging.getLogger(__name__)


async def stop_and_submit(
    job: StoppableJobProtocol,
    on_new_run: Callable[[str], None] | None = None,
) -> bool:
    """Stop job, submit new job, notify caller of new run_id. Returns True on success."""
    stop_result = await retry_async(
        job.stop,
        description="stop_job",
        max_retries=2,
    )

    if not stop_result.ok:
        try:
            status = await job.get_status()
        except Exception:
            logger.error("get_status_after_stop_failure_also_failed", exc_info=True)
            return False

        if status not in (JobStatus.STOPPED, JobStatus.FAILED):
            logger.error(
                "stop_job_failed_job_still_active status=%s, skipping submit",
                status.value,
            )
            return False

    try:
        run_id = await job.start()
    except Exception:
        logger.error("submit_job_failed", exc_info=True)
        return False

    if on_new_run is not None:
        on_new_run(run_id)
    return True


async def get_already_bad_nodes(node_manager: NodeManagerProtocol) -> set[str]:
    return set(await node_manager.get_bad_nodes())


async def retry_mark_node_bad(
    node_manager: NodeManagerProtocol,
    node_id: str,
    reason: str,
    node_metadata: dict[str, str] | None = None,
) -> RetryResult[None]:
    return await retry_async(
        lambda: node_manager.mark_node_bad(node_id, reason=reason, node_metadata=node_metadata),
        description=f"mark_node_bad({node_id})",
    )
