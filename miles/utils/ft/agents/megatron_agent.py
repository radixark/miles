from __future__ import annotations

import logging
import os
import socket
import time
from typing import Literal

from prometheus_client import Gauge

import miles.utils.ft.metric_names as mn
from miles.utils.ft.agents.controller_handle import ControllerHandleMixin
from miles.utils.ft.agents.prometheus_exporter import PrometheusExporter

logger = logging.getLogger(__name__)

_PHASE_TO_NUMERIC: dict[str, float] = {
    "idle": 0.0,
    "training": 1.0,
    "checkpoint_saving": 2.0,
}


class FtMegatronAgent(ControllerHandleMixin):
    """Embedded fault-tolerance agent for Megatron training processes.

    Each rank creates one instance. Exposes heartbeat gauges (iteration, phase)
    via a Prometheus HTTP exporter for the FtController to scrape.

    Training metrics (loss, grad_norm, etc.) are forwarded separately through
    FtTrackingAgent, which hooks into tracking_utils.log().
    """

    def __init__(self, rank: int, world_size: int) -> None:
        super().__init__()
        self._rank = rank
        self._world_size = world_size
        self._run_id: str = os.environ.get("FT_TRAINING_RUN_ID", "")
        self._node_id: str = socket.gethostname()

        self._exporter = PrometheusExporter()
        self._labels: dict[str, str] = {
            "rank": str(self._rank),
            "node_id": self._node_id,
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

        self._iteration_child = iteration_gauge.labels(**self._labels)
        self._phase_child = phase_gauge.labels(**self._labels)
        self._iteration_child.set(0)
        self._phase_child.set(_PHASE_TO_NUMERIC["idle"])

        self._register_rank()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def maybe_create(
        cls,
        rank: int,
        world_size: int,
        enabled: bool = True,
    ) -> FtMegatronAgent | None:
        if not enabled:
            return None
        try:
            return cls(rank=rank, world_size=world_size)
        except Exception:
            logger.warning("Failed to create FtMegatronAgent", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_exporter_address(self) -> str:
        return self._exporter.get_address()

    def step(
        self,
        iteration: int,
        phase: Literal["idle", "training", "checkpoint_saving"] = "training",
    ) -> None:
        try:
            self._iteration_child.set(iteration)
            self._phase_child.set(_PHASE_TO_NUMERIC.get(phase, 0.0))
        except Exception:
            logger.warning(
                "FtMegatronAgent.step() failed at iteration=%d", iteration,
                exc_info=True,
            )

    def shutdown(self) -> None:
        self._exporter.shutdown()

    # ------------------------------------------------------------------
    # Internal: controller communication
    # ------------------------------------------------------------------

    _REGISTER_MAX_ATTEMPTS = 3
    _REGISTER_RETRY_DELAY = 2.0

    def _register_rank(self) -> None:
        if not self._run_id:
            logger.info("No FT_TRAINING_RUN_ID set, skipping rank registration")
            return

        controller = self._get_controller_handle()
        if controller is None:
            logger.warning("Cannot register rank: controller not available")
            return

        import ray

        for attempt in range(self._REGISTER_MAX_ATTEMPTS):
            try:
                ray.get(
                    controller.register_rank.remote(
                        run_id=self._run_id,
                        rank=self._rank,
                        world_size=self._world_size,
                        node_id=self._node_id,
                        exporter_address=self.get_exporter_address(),
                    ),
                    timeout=10,
                )
                logger.info("Rank %d registered successfully", self._rank)
                return
            except Exception:
                if attempt < self._REGISTER_MAX_ATTEMPTS - 1:
                    time.sleep(self._REGISTER_RETRY_DELAY)
                else:
                    logger.warning(
                        "Failed to register rank %d after %d attempts",
                        self._rank,
                        self._REGISTER_MAX_ATTEMPTS,
                        exc_info=True,
                    )
