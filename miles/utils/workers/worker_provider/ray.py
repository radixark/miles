import asyncio
import logging
from functools import partial

import ray.actor

from miles.utils.workers.ray_worker_manager import RayWorkerManager, WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5.0


class RayWorkerProvider(BaseWorkerProvider):
    def __init__(self, worker_manager_handle: ray.actor.ActorHandle, poll_interval_seconds: float = 5.0):
        self._worker_manager_handle = worker_manager_handle
        self._poll_interval_seconds = poll_interval_seconds

    @classmethod
    def create(cls) -> "RayWorkerProvider":
        return cls(worker_manager_handle=RayWorkerManager.get_handle())

    def get_worker_infos(self, *, cell_id: str) -> list[WorkerInfo]:
        return ray.get(self._worker_manager_handle.get_worker_infos.remote(cell_id))

    async def get_addr(self, worker_name: str) -> HostAndPort:
        return (await self.get_addrs(worker_name=worker_name))["primary"]

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return await self._worker_manager_handle.get_worker_addrs.remote(worker_name)

    async def watch_cells(self, reconcile: ReconcileFn, *, spec_names: list[str]) -> StopWatchFn:
        seen_infos: dict[str, CellInfo] = {}
        # the initial sync must complete (and raise on failure) before the watch is considered established
        await self._poll_once(reconcile, seen_infos=seen_infos, spec_names=spec_names)
        task = asyncio.create_task(self._watch_loop(reconcile, seen_infos, spec_names=spec_names))
        return partial(_cancel_and_await_task, task)

    async def _watch_loop(
        self, reconcile: ReconcileFn, seen_infos: dict[str, CellInfo], *, spec_names: list[str]
    ) -> None:
        while True:
            await asyncio.sleep(self._poll_interval_seconds)
            try:
                await self._poll_once(reconcile, seen_infos=seen_infos, spec_names=spec_names)
            except Exception:
                logger.exception("Worker provider poll failed; retrying")

    async def _poll_once(
        self, reconcile: ReconcileFn, seen_infos: dict[str, CellInfo], *, spec_names: list[str]
    ) -> None:
        all_infos = await self._worker_manager_handle.get_cell_infos.remote(spec_names=spec_names)
        observed_infos: dict[str, CellInfo] = {cell_id: info for cell_id, info in all_infos.items() if info.alive}
        for cell_id in sorted(set(seen_infos) | set(observed_infos)):
            observed_info = observed_infos.get(cell_id)
            if seen_infos.get(cell_id) == observed_info:
                continue
            await reconcile(cell_id, observed_info)
            if observed_info is None:
                seen_infos.pop(cell_id, None)
            else:
                seen_infos[cell_id] = observed_info


async def _cancel_and_await_task(task) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
