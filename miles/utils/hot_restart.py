from __future__ import annotations

from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.rpc.client.misc import RETRYABLE_ERRORS, ServerRestartedError
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError


# ================================= constants ==================================

_WORKER_NOT_INITIALIZED_TIMEOUT_SECONDS = 1800.0
_WORKER_POLL_INTERVAL_SECONDS = 5.0
_WORKER_POLL_ATTEMPT_TIMEOUT_SECONDS = 120.0
_WORKER_READY_TIMEOUT_SECONDS = 60.0


# ============================== worker take-over ==============================


async def wait_until_worker_not_initialized(
    handle: BaseWorkerHandle, *, timeout: float = _WORKER_NOT_INITIALIZED_TIMEOUT_SECONDS
) -> None:
    async def attempt(remaining: float) -> None:
        await handle.wait_ready(timeout=_WORKER_READY_TIMEOUT_SECONDS, allow_server_uuid_change=True)
        if await handle.is_initialized():
            raise _StillInitializedError("the worker still answers that a previous script already initialized it")

    await retry_until_deadline(
        attempt,
        total_seconds=timeout,
        retry_on=(_StillInitializedError, WorkerUnreachableError, ServerRestartedError, *RETRYABLE_ERRORS),
        attempt_seconds=_WORKER_POLL_ATTEMPT_TIMEOUT_SECONDS,
        initial_delay=_WORKER_POLL_INTERVAL_SECONDS,
        backoff_factor=1.0,
        log_fields={"tag": "hot_restart"},
    )


class _StillInitializedError(Exception):
    pass
