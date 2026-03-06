from __future__ import annotations

import logging
from typing import Literal

from prometheus_client import Gauge

import miles.utils.ft.models.metric_names as mn
from miles.utils.ft.agents.utils.prometheus_exporter import PrometheusExporter
from miles.utils.ft.utils.graceful_degrade import graceful_degrade

logger = logging.getLogger(__name__)


class TrainingRankHeartbeat:
    """Prometheus heartbeat gauges for a single training rank.

    Exposes iteration and phase gauges via an HTTP exporter that the
    FtController can scrape.  This class owns the PrometheusExporter
    lifecycle and is agnostic to controller communication.
    """

    def __init__(self, rank: int, node_id: str) -> None:
        self._exporter = PrometheusExporter()

        labels: dict[str, str] = {
            "rank": str(rank),
            "node_id": node_id,
        }

        iteration_gauge = Gauge(
            mn.TRAINING_ITERATION,
            "Current training iteration",
            labelnames=["rank", "node_id"],
            registry=self._exporter.registry,
        )
        phase_gauge = Gauge(
            mn.TRAINING_PHASE,
            "Current training phase (0=idle, 1=training, 2=checkpoint_saving)",
            labelnames=["rank", "node_id"],
            registry=self._exporter.registry,
        )

        self._iteration_child = iteration_gauge.labels(**labels)
        self._phase_child = phase_gauge.labels(**labels)
        self._last_iteration: int = -1

        self._iteration_child.set(0)
        self._phase_child.set(mn.PHASE_TO_NUMERIC["idle"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_exporter_address(self) -> str:
        return self._exporter.get_address()

    @graceful_degrade()
    def set_phase(
        self,
        phase: Literal["idle", "training", "checkpoint_saving"],
    ) -> None:
        self._phase_child.set(mn.PHASE_TO_NUMERIC[phase])

    @graceful_degrade()
    def step(self, iteration: int) -> None:
        if iteration <= self._last_iteration:
            logger.warning(
                "TrainingRankHeartbeat.step() non-increasing iteration: got %d, last was %d",
                iteration,
                self._last_iteration,
            )
            return

        self._last_iteration = iteration
        self._iteration_child.set(self._last_iteration)

    def shutdown(self) -> None:
        self._exporter.shutdown()
