from __future__ import annotations

import asyncio
import logging

from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models import DiagnosticResult, MetricSample

logger = logging.getLogger(__name__)

_MetricKey = tuple[str, frozenset[str]]
_CounterValueKey = tuple[str, tuple[tuple[str, str], ...]]


class FtNodeAgent:
    def __init__(
        self,
        node_id: str,
        collectors: list[BaseCollector] | None = None,
        collect_interval_seconds: float | None = None,
    ) -> None:
        self._node_id = node_id
        self._collectors = collectors or []
        self._stopped = False

        if collect_interval_seconds is not None:
            for collector in self._collectors:
                collector.collect_interval = collect_interval_seconds

        self._registry = CollectorRegistry()
        self._gauges: dict[_MetricKey, Gauge] = {}
        self._counters: dict[_MetricKey, Counter] = {}
        self._counter_last_values: dict[_CounterValueKey, float] = {}

        httpd, _thread = start_http_server(port=0, registry=self._registry)
        self._httpd = httpd
        self._port: int = httpd.server_port
        self._collector_tasks: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_exporter_address(self) -> str:
        return f"http://localhost:{self._port}"

    async def start(self) -> None:
        if self._stopped or self._collector_tasks:
            return

        loop = asyncio.get_running_loop()
        for collector in self._collectors:
            task = loop.create_task(self._run_single_collector(collector))
            self._collector_tasks.append(task)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        for task in self._collector_tasks:
            task.cancel()
        await asyncio.gather(*self._collector_tasks, return_exceptions=True)
        self._collector_tasks.clear()

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

        self._httpd.shutdown()
        self._httpd.server_close()

    # ------------------------------------------------------------------
    # Stub methods (future milestones)
    # ------------------------------------------------------------------

    async def collect_logs(self) -> dict[str, str]:
        raise NotImplementedError(
            "collect_logs will be implemented in diag-framework milestone"
        )

    async def run_diagnostic(self, diagnostic_type: str) -> DiagnosticResult:
        raise NotImplementedError(
            "run_diagnostic will be implemented in diag-framework milestone"
        )

    async def cleanup_training_processes(self, training_job_id: str) -> None:
        raise NotImplementedError(
            "cleanup_training_processes will be implemented in recovery-basic milestone"
        )

    # ------------------------------------------------------------------
    # Per-collector background task
    # ------------------------------------------------------------------

    async def _run_single_collector(self, collector: BaseCollector) -> None:
        collector_name = type(collector).__name__
        while True:
            try:
                result = await collector.collect()
                self._update_exporter(result.metrics)
            except Exception:
                logger.warning(
                    "Collector %s failed on node %s",
                    collector_name,
                    self._node_id,
                    exc_info=True,
                )

            await asyncio.sleep(collector.collect_interval)

    # ------------------------------------------------------------------
    # Exporter update
    # ------------------------------------------------------------------

    def _update_exporter(self, metrics: list[MetricSample]) -> None:
        for sample in metrics:
            label_keys = frozenset(sample.labels.keys())

            if sample.metric_type == "counter":
                self._update_counter(sample, label_keys)
            else:
                self._update_gauge(sample, label_keys)

    def _update_gauge(self, sample: MetricSample, label_keys: frozenset[str]) -> None:
        key: _MetricKey = (sample.name, label_keys)
        gauge = self._gauges.get(key)
        if gauge is None:
            sorted_keys = sorted(label_keys)
            gauge = Gauge(
                sample.name,
                f"FT node metric: {sample.name}",
                labelnames=sorted_keys,
                registry=self._registry,
            )
            self._gauges[key] = gauge

        if sample.labels:
            gauge.labels(**sample.labels).set(sample.value)
        else:
            gauge.set(sample.value)

    def _update_counter(self, sample: MetricSample, label_keys: frozenset[str]) -> None:
        metric_key: _MetricKey = (sample.name, label_keys)
        counter = self._counters.get(metric_key)
        if counter is None:
            sorted_keys = sorted(label_keys)
            base_name = sample.name.removesuffix("_total")
            counter = Counter(
                base_name,
                f"FT node metric: {sample.name}",
                labelnames=sorted_keys,
                registry=self._registry,
            )
            self._counters[metric_key] = counter

        value_key: _CounterValueKey = (sample.name, tuple(sorted(sample.labels.items())))
        last = self._counter_last_values.get(value_key, 0.0)
        delta = sample.value - last
        if delta > 0:
            if sample.labels:
                counter.labels(**sample.labels).inc(delta)
            else:
                counter.inc(delta)
            self._counter_last_values[value_key] = sample.value
