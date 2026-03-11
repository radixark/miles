from __future__ import annotations

import logging

from prometheus_client import Gauge

from miles.utils.ft.agents.core.rollout.rollout_cell_agent import CellHealthResult
from miles.utils.ft.agents.metrics.prometheus_exporter import PrometheusExporter
from miles.utils.ft.controller.metrics.metric_names import ROLLOUT_CELL_ALIVE, ROLLOUT_ENGINE_ALIVE

logger = logging.getLogger(__name__)


class RolloutMetricsExporter:
    """Prometheus metric exporter for rollout health checks.

    Exposes per-cell and per-engine alive gauges via an HTTP exporter that
    the FtController can scrape.  Owns the PrometheusExporter lifecycle.
    """

    def __init__(self) -> None:
        self._exporter = PrometheusExporter()

        self._engine_alive = Gauge(
            ROLLOUT_ENGINE_ALIVE,
            "1=alive, 0=dead",
            labelnames=["cell_id", "engine_index"],
            registry=self._exporter.registry,
        )
        self._cell_alive = Gauge(
            ROLLOUT_CELL_ALIVE,
            "1=all engines alive, 0=any dead",
            labelnames=["cell_id"],
            registry=self._exporter.registry,
        )

    @property
    def registry(self) -> object:
        return self._exporter.registry

    @property
    def address(self) -> str:
        return self._exporter.get_address()

    def update(self, result: CellHealthResult) -> None:
        self._cell_alive.labels(cell_id=result.cell_id).set(
            1.0 if result.is_healthy else 0.0
        )

        dead_set = set(result.dead_engine_indices)
        for i in range(result.total_engines):
            self._engine_alive.labels(
                cell_id=result.cell_id, engine_index=str(i)
            ).set(0.0 if i in dead_set else 1.0)

    def shutdown(self) -> None:
        self._exporter.shutdown()
