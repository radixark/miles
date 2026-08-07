from __future__ import annotations

from collections.abc import Sequence

import ray.actor

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.cell_operations.ray import RayCellOperations
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.ray import RayWorkerProvider


class RayBackendCapability(BackendCapability):
    def __init__(self, *, worker_manager_handle: ray.actor.ActorHandle) -> None:
        self._worker_manager_handle = worker_manager_handle

    def dynamic_worker_provider(self, *, pool_ids: Sequence[str]) -> BaseWorkerProvider:
        return RayWorkerProvider(worker_manager_handle=self._worker_manager_handle, pool_ids=list(pool_ids))

    def static_worker_provider(self, *, pool_id: str) -> BaseWorkerProvider:
        return RayWorkerProvider(worker_manager_handle=self._worker_manager_handle)

    def cell_operations(self) -> BaseCellOperations:
        return RayCellOperations(worker_manager_handle=self._worker_manager_handle)
