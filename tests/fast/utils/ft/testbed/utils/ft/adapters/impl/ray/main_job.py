from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from miles.utils.ft.adapters.types import JobStatus, MainJobProtocol
from tests.fast.utils.ft.testbed.ray.actor_group import TestbedRayTrainGroup

logger = logging.getLogger(__name__)


class TestbedMainJob(MainJobProtocol):
    """MainJobProtocol backed by TestbedRayTrainGroup.

    start() launches spawn_actors as a background task and returns
    immediately, mimicking production MainJob where job submission is
    non-blocking. This ensures _activate_run() (which creates the
    TrainingRankRoster) runs before workers try to register.

    get_status() returns PENDING while the spawn task is in progress
    (workers not yet ready).  This prevents the controller from
    misinterpreting a pending restart as a failed one — the
    ``resolve_main_job_restart`` path keeps polling until the job
    transitions to RUNNING or FAILED.
    """

    def __init__(self, train_group: TestbedRayTrainGroup) -> None:
        self._train_group = train_group
        self._spawn_task: asyncio.Task[None] | None = None

    async def start(self) -> str:
        run_id = uuid4().hex[:8]
        logger.info("TestbedMainJob.start: run_id=%s", run_id)
        self._spawn_task = asyncio.create_task(
            self._spawn_with_error_log(run_id=run_id)
        )
        return run_id

    async def _spawn_with_error_log(self, run_id: str) -> None:
        try:
            await self._train_group.spawn_actors(run_id=run_id)
        except Exception:
            logger.error("spawn_actors failed for run_id=%s", run_id, exc_info=True)

    async def stop(self, timeout_seconds: int = 300) -> None:
        logger.info("TestbedMainJob.stop")
        await self._train_group.kill_all()

    async def get_status(self) -> JobStatus:
        if self._spawn_task is not None and not self._spawn_task.done():
            return JobStatus.PENDING

        alive = await self._train_group.all_alive()
        result = JobStatus.RUNNING if alive else JobStatus.FAILED
        return result
