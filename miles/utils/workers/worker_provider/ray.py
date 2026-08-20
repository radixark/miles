from functools import partial

import ray.actor

from miles.utils.workers.polling_reconcile_loop import PollingReconcileLoop
from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.utils import build_rpc_handle_of_worker_info
from miles.utils.workers.worker_spec import NamedHostAndPorts

POLL_INTERVAL_SECONDS = 5.0


class RayWorkerProvider(BaseWorkerProvider):
    def __init__(
        self,
        worker_manager_handle: ray.actor.ActorHandle,
        *,
        pool_ids: list[str] | None = None,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    ):
        self._worker_manager_handle = worker_manager_handle
        self._pool_ids = pool_ids
        self._poll_interval_seconds = poll_interval_seconds

    # TEMPORARY: this layer is not meant to serve a suspend the inference controller drives, deliberately
    # violated until the weight-update fault tolerance work removes the need
    async def stop_cells(self, *, cell_ids: list[str]) -> None:
        await self._worker_manager_handle.stop_cells.remote(cell_ids)

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        refs = [self._worker_manager_handle.get_worker_infos.remote(cell_id) for cell_id in cell_ids]
        return ray.get(refs)

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return await self._worker_manager_handle.get_worker_addrs.remote(worker_name)

    def get_handles_of_worker_infos(self, infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
        actor_handles = self._get_actor_handles([info for info in infos if info.worker_class is None])
        return {
            info.name: (
                RayWorkerHandle(actor_handles[info.name])
                if info.worker_class is None
                else build_rpc_handle_of_worker_info(info)
            )
            for info in infos
        }

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        pool_ids = self._watched_pool_ids()
        loop = PollingReconcileLoop(
            list_cells=partial(self._list_alive_cells, pool_ids=pool_ids),
            poll_interval_seconds=self._poll_interval_seconds,
        )
        return await loop.start(reconcile)

    async def _list_alive_cells(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        all_infos = await self._worker_manager_handle.get_cell_infos.remote(pool_ids=pool_ids)
        return {cell_id: info for cell_id, info in all_infos.items() if info.alive}

    def _get_actor_handles(self, infos: list[WorkerInfo]) -> dict[str, ray.actor.ActorHandle]:
        refs = [
            self._worker_manager_handle.get_actor_handle.remote(info.name, expected_generation=info.generation)
            for info in infos
        ]
        return {info.name: handle for info, handle in zip(infos, ray.get(refs), strict=True)}

    def _watched_pool_ids(self) -> list[str]:
        assert self._pool_ids is not None, "this provider was built without the pool_ids it is meant to observe"
        return self._pool_ids
