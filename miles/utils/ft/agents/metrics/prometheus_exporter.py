from __future__ import annotations

import logging
import threading
import time
import urllib.error
import urllib.request
from typing import TypeVar

from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

from miles.utils.ft.agents.types import CounterSample, GaugeSample, _MetricSampleBase
from miles.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)

_MetricKey = tuple[str, frozenset[str]]
_M = TypeVar("_M", Gauge, Counter)


class PrometheusExporter:
    """Manages a Prometheus CollectorRegistry with an HTTP server for metric exposition.

    Used by both FtNodeAgent (dynamic metrics from collectors) and
    FtTrainingRankAgent (pre-defined heartbeat gauges).
    """

    def __init__(self) -> None:
        self._registry = CollectorRegistry()
        self._gauges: dict[_MetricKey, Gauge] = {}
        self._counters: dict[_MetricKey, Counter] = {}

        # The HTTP server runs in a background thread and may scrape the registry
        # concurrently with collector tasks calling update_metrics(). A lock
        # serializes all mutations to the gauge/counter caches and the registry
        # so that the scrape thread never observes a partially-registered metric.
        self._lock = threading.RLock()

        httpd, _thread = start_http_server(port=0, registry=self._registry)
        self._httpd = httpd
        self.port: int = httpd.server_port
        logger.info("metrics: prometheus exporter started: port=%d", self.port)

    @property
    def registry(self) -> CollectorRegistry:
        return self._registry

    def get_address(self) -> str:
        _hostname, ip = get_host_info()
        return f"http://{ip}:{self.port}"

    def wait_until_ready(
        self,
        timeout_seconds: float = 5.0,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        metrics_url = f"{self.get_address()}/metrics"
        last_exception: Exception | None = None

        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(metrics_url, timeout=poll_interval_seconds):
                    return
            except Exception as exc:
                last_exception = exc
                time.sleep(poll_interval_seconds)

        logger.error("metrics: prometheus exporter not ready after timeout: url=%s, timeout=%s", metrics_url, timeout_seconds)
        raise RuntimeError(f"Prometheus exporter did not become ready: {metrics_url}") from last_exception

    def shutdown(self) -> None:
        logger.info("metrics: prometheus exporter shutting down: port=%d", self.port)
        with self._lock:
            self._httpd.shutdown()
            self._httpd.server_close()

    def update_metrics(self, metrics: list[GaugeSample | CounterSample]) -> None:
        with self._lock:
            for sample in metrics:
                match sample:
                    case GaugeSample():
                        self._resolve_child(self._gauges, Gauge, sample).set(sample.value)
                    case CounterSample() if sample.delta > 0:
                        self._resolve_child(self._counters, Counter, sample).inc(sample.delta)
                    case CounterSample():
                        logger.debug(
                            "metrics: counter delta non-positive, skipping: name=%s, delta=%s",
                            sample.name,
                            sample.delta,
                        )

    def _resolve_child(
        self,
        cache: dict[_MetricKey, _M],
        metric_cls: type[_M],
        sample: _MetricSampleBase,
    ) -> Gauge | Counter:
        metric = self._get_or_create(cache, metric_cls, sample)
        return metric.labels(**sample.labels) if sample.labels else metric

    def _get_or_create(
        self,
        cache: dict[_MetricKey, _M],
        metric_cls: type[_M],
        sample: _MetricSampleBase,
    ) -> _M:
        label_keys = frozenset(sample.labels.keys())
        key: _MetricKey = (sample.name, label_keys)
        metric = cache.get(key)

        if metric is not None:
            return metric

        name = sample.name.removesuffix("_total") if metric_cls is Counter else sample.name
        metric = metric_cls(
            name,
            f"FT node metric: {sample.name}",
            labelnames=sorted(label_keys),
            registry=self._registry,
        )
        cache[key] = metric
        return metric
