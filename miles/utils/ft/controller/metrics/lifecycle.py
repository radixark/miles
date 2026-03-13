from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from miles.utils.ft.controller.types import TimeSeriesStoreLifecycle

logger = logging.getLogger(__name__)

_SCRAPE_RESTART_DELAY_SECONDS = 10.0
_MAX_SCRAPE_RESTARTS = 20


@dataclass
class MetricStoreTaskHandle:
    task: asyncio.Task[None] = field(init=False)
    last_exception: Exception | None = None

    @property
    def is_unhealthy(self) -> bool:
        return self.task.done()

    def format_health_error(self) -> str:
        parts = [f"task_done={self.task.done()}"]
        if self.last_exception is not None:
            parts.append(f"last_exception={type(self.last_exception).__name__}: {self.last_exception}")
        return "Metric store unhealthy: " + ", ".join(parts)


async def start_metric_store_task(store: TimeSeriesStoreLifecycle) -> MetricStoreTaskHandle:
    handle = MetricStoreTaskHandle()

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
                logger.error(
                    "metric_store_crashed restart=%d/%d, restarting in %.0fs",
                    restarts,
                    _MAX_SCRAPE_RESTARTS,
                    _SCRAPE_RESTART_DELAY_SECONDS,
                    exc_info=True,
                )
                await asyncio.sleep(_SCRAPE_RESTART_DELAY_SECONDS)

        logger.error("metric_store_exhausted_restarts max=%d", _MAX_SCRAPE_RESTARTS)

    handle.task = asyncio.create_task(_run())
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
