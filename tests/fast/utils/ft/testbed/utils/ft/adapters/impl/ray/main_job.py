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
    """

    def __init__(self, train_group: TestbedRayTrainGroup) -> None:
        self._train_group = train_group
        self._spawn_task: asyncio.Task[None] | None = None

    async def start(self) -> str:
        run_id = uuid4().hex[:8]
        self._spawn_task = asyncio.create_task(
            self._spawn_with_error_log(run_id=run_id)
        )
        return run_id

    async def _spawn_with_error_log(self, run_id: str) -> None:
        try:
            await self._train_group.spawn_actors(run_id=run_id)
        except Exception:
            logger.error("spawn_actors failed", exc_info=True)

    async def stop(self, timeout_seconds: int = 300) -> None:
        await self._train_group.kill_all()

    async def get_status(self) -> JobStatus:
        if await self._train_group.all_alive():
            return JobStatus.RUNNING
        return JobStatus.FAILED
