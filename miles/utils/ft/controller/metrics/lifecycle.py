from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.controller.types import TimeSeriesStoreLifecycle

logger = logging.getLogger(__name__)

_SCRAPE_RESTART_DELAY_SECONDS = 10.0


async def start_metric_store_task(store: TimeSeriesStoreLifecycle) -> asyncio.Task[None]:
    async def _run() -> None:
        while True:
            try:
                await store.start()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.error(
                    "scrape_loop_crashed, restarting in %.0fs",
                    _SCRAPE_RESTART_DELAY_SECONDS,
                    exc_info=True,
                )
                await asyncio.sleep(_SCRAPE_RESTART_DELAY_SECONDS)

    task = asyncio.create_task(_run())
    logger.info("scrape_loop_started")
    return task


async def stop_metric_store_task(
    store: TimeSeriesStoreLifecycle,
    task: asyncio.Task[None],
) -> None:
    try:
        await store.stop()
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    logger.info("scrape_loop_stopped")
