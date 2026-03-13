from __future__ import annotations

import logging
from typing import Literal

from prometheus_client import Gauge

import miles.utils.ft.utils.metric_names as mn
from miles.utils.ft.agents.metrics.prometheus_exporter import PrometheusExporter
from miles.utils.ft.utils.graceful_degrade import graceful_degrade

logger = logging.getLogger(__name__)


class TrainingRankExporter:
    """Prometheus metric exporter for a single training rank.

    Exposes heartbeat and phase gauges via an HTTP exporter that the
    FtController can scrape.  This class owns the PrometheusExporter
    lifecycle and is agnostic to controller communication.
    """

    def __init__(self, rank: int, node_id: str, run_id: str) -> None:
        if not run_id:
            logger.error("metrics: TrainingRankExporter requires non-empty run_id: rank=%d, node_id=%s", rank, node_id)
            raise ValueError("run_id must not be empty for TrainingRankExporter")

        self._exporter = PrometheusExporter()

        labels: dict[str, str] = {
            "rank": str(rank),
            "node_id": node_id,
            "ft_run_id": run_id,
        }

        heartbeat_gauge = Gauge(
            mn.AGENT_HEARTBEAT,
            "Agent heartbeat counter (monotonically increasing)",
            labelnames=["rank", "node_id", "ft_run_id"],
            registry=self._exporter.registry,
        )
        phase_gauge = Gauge(
            mn.TRAINING_PHASE,
            "Current training phase (0=idle, 1=training, 2=checkpoint_saving)",
            labelnames=["rank", "node_id", "ft_run_id"],
            registry=self._exporter.registry,
        )

        self._heartbeat_child = heartbeat_gauge.labels(**labels)
        self._phase_child = phase_gauge.labels(**labels)
        self._heartbeat_counter: int = 0

        self._heartbeat_child.set(0)
        self._phase_child.set(mn.PHASE_TO_NUMERIC["idle"])
        logger.info(
            "metrics: training rank exporter initialized: rank=%d, node_id=%s, run_id=%s, address=%s",
            rank,
            node_id,
            run_id,
            self._exporter.get_address(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_exporter_address(self) -> str:
        return self._exporter.get_address()

    def wait_until_ready(self, timeout_seconds: float = 5.0) -> None:
        self._exporter.wait_until_ready(timeout_seconds=timeout_seconds)

    @graceful_degrade()
    def set_phase(
        self,
        phase: Literal["idle", "training", "checkpoint_saving"],
    ) -> None:
        self._phase_child.set(mn.PHASE_TO_NUMERIC[phase])
        self._bump_heartbeat()

    @graceful_degrade()
    def step(self) -> None:
        self._bump_heartbeat()

    def shutdown(self) -> None:
        logger.info("metrics: training rank exporter shutting down")
        self._exporter.shutdown()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _bump_heartbeat(self) -> None:
        self._heartbeat_counter += 1
        self._heartbeat_child.set(self._heartbeat_counter)
