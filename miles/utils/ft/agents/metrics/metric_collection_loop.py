from __future__ import annotations

import asyncio
import logging
import time

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.metrics.prometheus_exporter import PrometheusExporter
from miles.utils.ft.agents.types import CounterSample, GaugeSample, MetricSample

logger = logging.getLogger(__name__)

_STALENESS_METRIC = "ft_collector_last_success_timestamp"
_CONSECUTIVE_FAILURES_METRIC = "ft_collector_consecutive_failures"


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
        self._last_success_timestamps: dict[str, float] = {}

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
        consecutive_failures = 0

        while True:
            try:
                result = await collector.collect()
                labeled = _inject_node_id(result.metrics, node_id=self._node_id)
                self._exporter.update_metrics(labeled)
                consecutive_failures = 0
                self._last_success_timestamps[collector_name] = time.time()
            except Exception:
                consecutive_failures += 1
                logger.warning(
                    "Collector %s failed on node %s (consecutive_failures=%d)",
                    collector_name,
                    self._node_id,
                    consecutive_failures,
                    exc_info=True,
                )

            self._emit_staleness_metrics(
                collector_name=collector_name,
                consecutive_failures=consecutive_failures,
            )

            await asyncio.sleep(collector.collect_interval)

    def _emit_staleness_metrics(
        self,
        *,
        collector_name: str,
        consecutive_failures: int,
    ) -> None:
        labels = {"collector": collector_name, "node_id": self._node_id}
        last_success = self._last_success_timestamps.get(collector_name, 0.0)
        self._exporter.update_metrics(
            [
                GaugeSample(name=_STALENESS_METRIC, labels=labels, value=last_success),
                GaugeSample(
                    name=_CONSECUTIVE_FAILURES_METRIC,
                    labels=labels,
                    value=float(consecutive_failures),
                ),
            ]
        )


def _inject_node_id(
    samples: list[MetricSample],
    *,
    node_id: str,
) -> list[GaugeSample | CounterSample]:
    """Merge node_id into every sample's labels.

    Collector-returned samples typically lack node_id because collectors
    don't know their placement. This ensures all node-agent metrics carry
    node_id regardless of backend (MiniPrometheus auto-injects via
    target_id, but real Prometheus does not).
    """
    result: list[GaugeSample | CounterSample] = []
    for sample in samples:
        if "node_id" not in sample.labels:
            merged_labels = {**sample.labels, "node_id": node_id}
            sample = sample.model_copy(update={"labels": merged_labels})
        result.append(sample)
    return result
