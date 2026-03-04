from __future__ import annotations

import logging
from typing import TypeVar

from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

from miles.utils.ft.models import MetricSample

logger = logging.getLogger(__name__)

_MetricKey = tuple[str, frozenset[str]]
_M = TypeVar("_M", Gauge, Counter)


class PrometheusExporter:
    """Manages a Prometheus CollectorRegistry with an HTTP server for metric exposition.

    Used by both FtNodeAgent (dynamic metrics from collectors) and
    FtMegatronAgent (pre-defined heartbeat gauges).
    """

    def __init__(self) -> None:
        self._registry = CollectorRegistry()
        self._gauges: dict[_MetricKey, Gauge] = {}
        self._counters: dict[_MetricKey, Counter] = {}

        httpd, _thread = start_http_server(port=0, registry=self._registry)
        self._httpd = httpd
        self.port: int = httpd.server_port

    @property
    def registry(self) -> CollectorRegistry:
        return self._registry

    def get_address(self) -> str:
        return f"http://localhost:{self.port}"

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def update_metrics(self, metrics: list[MetricSample]) -> None:
        for sample in metrics:
            if sample.metric_type == "counter":
                self._update_counter(sample)
            else:
                self._update_gauge(sample)

    def _update_gauge(self, sample: MetricSample) -> None:
        gauge = self._get_or_create(self._gauges, Gauge, sample)

        child = gauge.labels(**sample.labels) if sample.labels else gauge
        child.set(sample.value)

    def _update_counter(self, sample: MetricSample) -> None:
        counter = self._get_or_create(self._counters, Counter, sample)

        if sample.value > 0:
            child = counter.labels(**sample.labels) if sample.labels else counter
            child.inc(sample.value)

    def _get_or_create(
        self,
        cache: dict[_MetricKey, _M],
        metric_cls: type[_M],
        sample: MetricSample,
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
