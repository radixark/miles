import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import partial

from miles.utils.workers.worker_provider.base import CellInfo, CellReconcileFn, StopWatchFn

logger = logging.getLogger(__name__)

ListCellsFn = Callable[[], Awaitable[dict[str, CellInfo]]]


class PollingReconcileLoop:
    def __init__(self, *, list_cells: ListCellsFn, poll_interval_seconds: float) -> None:
        self._list_cells = list_cells
        self._poll_interval_seconds = poll_interval_seconds

    async def start(self, reconcile: CellReconcileFn) -> StopWatchFn:
        seen_infos: dict[str, CellInfo] = {}
        # the initial sync must complete (and raise on failure) before the watch is considered established
        await self._poll_once(reconcile, seen_infos=seen_infos)
        task = asyncio.create_task(self._watch_loop(reconcile, seen_infos))
        return partial(cancel_and_await_task, task)

    async def _watch_loop(self, reconcile: CellReconcileFn, seen_infos: dict[str, CellInfo]) -> None:
        while True:
            await asyncio.sleep(self._poll_interval_seconds)
            try:
                await self._poll_once(reconcile, seen_infos=seen_infos)
            except Exception:
                logger.exception("Worker provider poll failed; retrying")

    async def _poll_once(self, reconcile: CellReconcileFn, *, seen_infos: dict[str, CellInfo]) -> None:
        observed_infos = await self._list_cells()
        for cell_id in sorted(set(seen_infos) | set(observed_infos)):
            observed_info = observed_infos.get(cell_id)
            if seen_infos.get(cell_id) == observed_info:
                continue
            await reconcile(cell_id, observed_info)
            if observed_info is None:
                seen_infos.pop(cell_id, None)
            else:
                seen_infos[cell_id] = observed_info


async def cancel_and_await_task(task) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
