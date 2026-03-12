from __future__ import annotations

import logging

from prometheus_client import Gauge

from miles.utils.ft.agents.metrics.prometheus_exporter import PrometheusExporter
from miles.utils.ft.utils.metric_names import ROLLOUT_CELL_ALIVE

logger = logging.getLogger(__name__)


class RolloutMetricsExporter:
    """Prometheus metric exporter for rollout health checks.

    Exposes a per-cell alive gauge via an HTTP exporter that the
    FtController can scrape.  Owns the PrometheusExporter lifecycle.
    """

    def __init__(self) -> None:
        self._exporter = PrometheusExporter()

        self._cell_alive = Gauge(
            ROLLOUT_CELL_ALIVE,
            "1=cell alive, 0=cell dead",
            labelnames=["cell_id"],
            registry=self._exporter.registry,
        )

    @property
    def registry(self) -> object:
        return self._exporter.registry

    @property
    def address(self) -> str:
        return self._exporter.get_address()

    def update(self, *, cell_id: str, is_healthy: bool) -> None:
        self._cell_alive.labels(cell_id=cell_id).set(float(is_healthy))

    def shutdown(self) -> None:
        self._exporter.shutdown()
