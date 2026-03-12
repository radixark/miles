from __future__ import annotations

from uuid import uuid4

from miles.utils.ft.adapters.types import JobStatus, MainJobProtocol
from tests.fast.utils.ft.testbed.ray.actor_group import TestbedRayTrainGroup


class TestbedMainJob(MainJobProtocol):
    """MainJobProtocol backed by TestbedRayTrainGroup.

    start() spawns real Ray actors pinned to training nodes.
    get_status() pings all workers — dead workers mean FAILED.
    """

    def __init__(self, train_group: TestbedRayTrainGroup) -> None:
        self._train_group = train_group

    async def start(self) -> str:
        run_id = uuid4().hex[:8]
        await self._train_group.spawn_actors(run_id=run_id)
        return run_id

    async def stop(self, timeout_seconds: int = 300) -> None:
        self._train_group.kill_all()

    async def get_status(self) -> JobStatus:
        if await self._train_group.all_alive():
            return JobStatus.RUNNING
        return JobStatus.FAILED
