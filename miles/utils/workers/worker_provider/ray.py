import logging

import ray.actor

from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5.0


class RayWorkerProvider(BaseWorkerProvider):
    def __init__(self, worker_manager_handle: ray.actor.ActorHandle):
        self._worker_manager_handle = worker_manager_handle

    @classmethod
    def create(cls) -> "RayWorkerProvider":
        return cls(worker_manager_handle=RayWorkerManager.get_handle())

    async def get_addr(self, worker_name: str) -> HostAndPort:
        return await self._worker_manager_handle.get_worker_addr.remote(worker_name)
