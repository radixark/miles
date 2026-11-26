from __future__ import annotations

import ray.actor

from miles.utils.workers.worker_manager.client import ManagerBackendCapability
from miles.utils.workers.worker_provider.ray import RayWorkerManagerClient


class RayBackendCapability(ManagerBackendCapability):
    def __init__(self, *, worker_manager_handle: ray.actor.ActorHandle) -> None:
        super().__init__(RayWorkerManagerClient(worker_manager_handle))
