from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.protocols.metrics import (
    MetricQueryProtocol,
    MetricStoreLifecycle,
    MetricStoreProtocol,
    ScrapeTargetManagerProtocol,
)

__all__ = [
    "MetricQueryProtocol",
    "MetricStoreLifecycle",
    "MetricStoreProtocol",
    "ScrapeTargetManagerProtocol",
    "start_metric_store_task",
    "stop_metric_store_task",
]

logger = logging.getLogger(__name__)


async def start_metric_store_task(store: MetricStoreLifecycle) -> asyncio.Task[None]:
    async def _run() -> None:
        try:
            await store.start()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("scrape_loop_crashed", exc_info=True)

    task = asyncio.create_task(_run())
    logger.info("scrape_loop_started")
    return task


async def stop_metric_store_task(
    store: MetricStoreLifecycle, task: asyncio.Task[None],
) -> None:
    await store.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("scrape_loop_stopped")
