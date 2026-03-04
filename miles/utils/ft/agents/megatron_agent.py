from __future__ import annotations

import logging
import os
import socket
from typing import Any

from prometheus_client import CollectorRegistry, Gauge, start_http_server

logger = logging.getLogger(__name__)

_PHASE_TO_NUMERIC: dict[str, float] = {
    "idle": 0.0,
    "training": 1.0,
    "checkpoint_saving": 2.0,
}


class FtMegatronAgent:
    """Embedded fault-tolerance agent for Megatron training processes.

    Each rank creates one instance. Exposes heartbeat gauges via a Prometheus
    HTTP exporter and pushes per-step metrics to FtController (fire-and-forget).
    """

    def __init__(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size
        self._run_id: str = os.environ.get("FT_TRAINING_RUN_ID", "")
        self._node_id: str = socket.gethostname()

        self._registry = CollectorRegistry()
        self._iteration_gauge = Gauge(
            "training_iteration",
            "Current training iteration",
            labelnames=["rank", "node_id"],
            registry=self._registry,
        )
        self._phase_gauge = Gauge(
            "training_phase",
            "Current training phase (0=idle, 1=training, 2=checkpoint_saving)",
            labelnames=["rank", "node_id"],
            registry=self._registry,
        )

        self._iteration_gauge.labels(
            rank=str(self._rank), node_id=self._node_id
        ).set(0)
        self._phase_gauge.labels(
            rank=str(self._rank), node_id=self._node_id
        ).set(_PHASE_TO_NUMERIC["idle"])

        self._controller_handle: Any | None = None

        httpd, _thread = start_http_server(port=0, registry=self._registry)
        self._httpd = httpd
        self._port: int = httpd.server_port

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_exporter_address(self) -> str:
        return f"http://localhost:{self._port}"

    def step(
        self,
        iteration: int,
        loss: float | None = None,
        grad_norm: float | None = None,
        mfu: float | None = None,
        iteration_time: float | None = None,
        phase: str = "training",
    ) -> None:
        self._iteration_gauge.labels(
            rank=str(self._rank), node_id=self._node_id
        ).set(iteration)
        self._phase_gauge.labels(
            rank=str(self._rank), node_id=self._node_id
        ).set(_PHASE_TO_NUMERIC.get(phase, 0.0))

        metrics: dict[str, float] = {}
        if loss is not None:
            metrics["loss"] = loss
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm
        if mfu is not None:
            metrics["mfu"] = mfu
        if iteration_time is not None:
            metrics["iteration_time"] = iteration_time

        if metrics and self._run_id:
            controller = self._get_controller_handle()
            if controller is not None:
                controller.log_step.remote(
                    run_id=self._run_id,
                    rank=self._rank,
                    step=iteration,
                    metrics=metrics,
                )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_controller_handle(self) -> Any | None:
        if self._controller_handle is not None:
            return self._controller_handle

        try:
            import ray

            self._controller_handle = ray.get_actor("ft_controller")
        except Exception:
            logger.warning("Failed to get ft_controller actor handle")
            return None

        return self._controller_handle
