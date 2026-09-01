import asyncio
import logging
from functools import partial

import ray.actor

from miles.utils.misc import cancel_and_await_task
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import NamedHostAndPorts

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5.0


class RayWorkerProvider(BaseWorkerProvider):
    def __init__(
        self,
        worker_manager_handle: ray.actor.ActorHandle,
        *,
        pool_ids: list[str] | None = None,
        poll_interval_seconds: float = 5.0,
    ):
        self._worker_manager_handle = worker_manager_handle
        self._pool_ids = pool_ids
        self._poll_interval_seconds = poll_interval_seconds

    @classmethod
    def create(cls, *, pool_ids: list[str] | None = None) -> "RayWorkerProvider":
        return cls(worker_manager_handle=RayWorkerManager.get_handle(), pool_ids=pool_ids)

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return await self._worker_manager_handle.get_worker_addrs.remote(worker_name)

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        pool_ids = self._watched_pool_ids()
        seen_infos: dict[str, CellInfo] = {}
        # the initial sync must complete (and raise on failure) before the watch is considered established
        await self._poll_once(reconcile, seen_infos=seen_infos, pool_ids=pool_ids)
        task = asyncio.create_task(self._watch_loop(reconcile, seen_infos, pool_ids=pool_ids))
        return partial(cancel_and_await_task, task)

    def _watched_pool_ids(self) -> list[str]:
        assert self._pool_ids is not None, "this provider was built without the pool_ids it is meant to observe"
        return self._pool_ids

    async def _watch_loop(
        self, reconcile: ReconcileFn, seen_infos: dict[str, CellInfo], *, pool_ids: list[str]
    ) -> None:
        while True:
            await asyncio.sleep(self._poll_interval_seconds)
            try:
                await self._poll_once(reconcile, seen_infos=seen_infos, pool_ids=pool_ids)
            except Exception:
                logger.exception("Worker provider poll failed; retrying")

    async def _poll_once(
        self, reconcile: ReconcileFn, seen_infos: dict[str, CellInfo], *, pool_ids: list[str]
    ) -> None:
        all_infos = await self._worker_manager_handle.get_cell_infos.remote(pool_ids=pool_ids)
        observed_infos: dict[str, CellInfo] = {cell_id: info for cell_id, info in all_infos.items() if info.alive}
        first_error: Exception | None = None
        for cell_id in sorted(set(seen_infos) | set(observed_infos)):
            observed_info = observed_infos.get(cell_id)
            if seen_infos.get(cell_id) == observed_info:
                continue
            try:
                await reconcile(cell_id, observed_info)
            except Exception as error:
                logger.exception(f"Reconciling cell {cell_id} failed; continuing with the other cells")
                first_error = first_error or error
                continue
            if observed_info is None:
                seen_infos.pop(cell_id, None)
            else:
                seen_infos[cell_id] = observed_info

        if first_error is not None:
            raise first_error
