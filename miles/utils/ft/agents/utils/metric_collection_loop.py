from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.utils.prometheus_exporter import PrometheusExporter

logger = logging.getLogger(__name__)


class MetricCollectionLoop:
    """Runs collectors as background tasks and feeds metrics to a PrometheusExporter."""

    def __init__(
        self,
        node_id: str,
        collectors: list[BaseCollector],
        exporter: PrometheusExporter,
    ) -> None:
        self._node_id = node_id
        self._collectors = collectors
        self._exporter = exporter
        self._tasks: list[asyncio.Task[None]] = []
        self._stopped = False

    @property
    def tasks(self) -> list[asyncio.Task[None]]:
        return self._tasks

    async def start(self) -> None:
        if self._stopped or self._tasks:
            return

        loop = asyncio.get_running_loop()
        for collector in self._collectors:
            task = loop.create_task(self._run_single_collector(collector))
            self._tasks.append(task)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for collector in self._collectors:
            try:
                await collector.close()
            except Exception:
                logger.warning(
                    "Collector %s.close() failed on node %s",
                    type(collector).__name__,
                    self._node_id,
                    exc_info=True,
                )

    async def _run_single_collector(self, collector: BaseCollector) -> None:
        collector_name = type(collector).__name__
        while True:
            try:
                result = await collector.collect()
                self._exporter.update_metrics(result.metrics)
            except Exception:
                logger.warning(
                    "Collector %s failed on node %s",
                    collector_name,
                    self._node_id,
                    exc_info=True,
                )

            await asyncio.sleep(collector.collect_interval)
