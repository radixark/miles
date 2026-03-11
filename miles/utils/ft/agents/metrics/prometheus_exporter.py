from __future__ import annotations

import logging
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

        httpd, _thread = start_http_server(port=0, registry=self._registry)
        self._httpd = httpd
        self.port: int = httpd.server_port

    @property
    def registry(self) -> CollectorRegistry:
        return self._registry

    def get_address(self) -> str:
        _hostname, ip = get_host_info()
        return f"http://{ip}:{self.port}"

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def update_metrics(self, metrics: list[GaugeSample | CounterSample]) -> None:
        for sample in metrics:
            match sample:
                case GaugeSample():
                    self._resolve_child(self._gauges, Gauge, sample).set(sample.value)
                case CounterSample() if sample.delta > 0:
                    self._resolve_child(self._counters, Counter, sample).inc(sample.delta)

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
