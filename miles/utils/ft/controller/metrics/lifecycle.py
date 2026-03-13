from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from miles.utils.ft.controller.types import TimeSeriesStoreLifecycle

logger = logging.getLogger(__name__)

_SCRAPE_RESTART_DELAY_SECONDS = 10.0
_MAX_SCRAPE_RESTARTS = 20


@dataclass
class MetricStoreTaskHandle:
    task: asyncio.Task[None]
    restart_exhausted: bool = False
    last_exception: Exception | None = None
    last_failure_at: float | None = None


def is_metric_store_unhealthy(handle: MetricStoreTaskHandle) -> bool:
    return handle.task.done() or handle.restart_exhausted


def format_metric_store_health_error(handle: MetricStoreTaskHandle) -> str:
    parts = [
        f"task_done={handle.task.done()}",
        f"restart_exhausted={handle.restart_exhausted}",
    ]
    if handle.last_exception is not None:
        parts.append(f"last_exception={type(handle.last_exception).__name__}: {handle.last_exception}")
    if handle.last_failure_at is not None:
        parts.append(f"last_failure_at={handle.last_failure_at:.1f}")
    return "Metric store unhealthy: " + ", ".join(parts)


async def start_metric_store_task(store: TimeSeriesStoreLifecycle) -> MetricStoreTaskHandle:
    handle = MetricStoreTaskHandle(task=asyncio.Future())  # placeholder, replaced below

    async def _run() -> None:
        restarts = 0
        while restarts < _MAX_SCRAPE_RESTARTS:
            try:
                await store.start()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                restarts += 1
                handle.last_exception = exc
                handle.last_failure_at = time.monotonic()
                logger.error(
                    "metric_store_crashed restart=%d/%d, restarting in %.0fs",
                    restarts,
                    _MAX_SCRAPE_RESTARTS,
                    _SCRAPE_RESTART_DELAY_SECONDS,
                    exc_info=True,
                )
                await asyncio.sleep(_SCRAPE_RESTART_DELAY_SECONDS)

        handle.restart_exhausted = True
        logger.error("metric_store_exhausted_restarts max=%d", _MAX_SCRAPE_RESTARTS)

    task = asyncio.create_task(_run())
    handle.task = task
    logger.info("metric_store_started")
    return handle


async def stop_metric_store_task(
    store: TimeSeriesStoreLifecycle,
    handle: MetricStoreTaskHandle,
) -> None:
    try:
        await store.stop()
    finally:
        handle.task.cancel()
        try:
            await handle.task
        except asyncio.CancelledError:
            pass
    logger.info("metric_store_stopped")
