from __future__ import annotations

import asyncio
import logging
from argparse import Namespace
from typing import Any

from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.rpc.client.misc import RETRYABLE_ERRORS, ServerRestartedError
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError

logger = logging.getLogger(__name__)


# ================================= constants ==================================

TAKE_OVER_GATE_TIMEOUT_SECONDS = 600.0
_TRAINER_RELOAD_TIMEOUT_SECONDS = 3600.0
_INFERENCE_IDLE_TIMEOUT_SECONDS = 3600.0
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


# ============================= trainer take-over ==============================


async def wait_trainers_idle(handles: dict[str, BaseWorkerHandle]) -> bool:
    assert handles

    initialized = await asyncio.gather(*[handle.is_initialized() for handle in handles.values()])
    assert len(set(initialized)) == 1, f"trainers disagree about being initialized: {list(initialized)}"
    [resumed] = set(initialized)

    if resumed:
        for trainer_id, handle in handles.items():
            logger.info(f"Waiting until trainer {trainer_id!r} finished the call the previous script left running")
            await handle.wait_idle(timeout=TAKE_OVER_GATE_TIMEOUT_SECONDS)

    return resumed


async def trainer_init_or_load_state(
    trainer: BaseWorkerHandle, model_args: Namespace, *, trainer_id: str, resumed: bool
) -> list[Any]:
    if not resumed:
        return await trainer.init(model_args)

    start_rollout_ids = await asyncio.wait_for(trainer.load_state(), timeout=_TRAINER_RELOAD_TIMEOUT_SECONDS)
    logger.info(f"Resumed the already-initialized trainer {trainer_id!r} at rollout ids {start_rollout_ids}")
    return start_rollout_ids


# ============================ inference take-over =============================


async def init_or_reset_inference_controller(inference_controller: BaseWorkerHandle) -> None:
    if not await inference_controller.is_initialized():
        await inference_controller.init()
        return

    logger.info("The inference controller outlived a previous orchestration script; taking it over as it is")

    await inference_controller.wait_idle(timeout=_INFERENCE_IDLE_TIMEOUT_SECONDS)

    await inference_controller.wait_expected_num_cells(timeout=TAKE_OVER_GATE_TIMEOUT_SECONDS)

    await asyncio.wait_for(inference_controller.abort_all(), timeout=TAKE_OVER_GATE_TIMEOUT_SECONDS)
    logger.info("Asked every engine of the fleet to abort the generations it was still running")
