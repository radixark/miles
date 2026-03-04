from __future__ import annotations

import logging
import os
import socket
import time
from typing import Any, Literal

from prometheus_client import CollectorRegistry, Gauge, start_http_server

import miles.utils.ft.metric_names as mn

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
        self._labels: dict[str, str] = {
            "rank": str(self._rank),
            "node_id": self._node_id,
        }

        iteration_gauge = Gauge(
            mn.TRAINING_ITERATION,
            "Current training iteration",
            labelnames=["rank", "node_id"],
            registry=self._registry,
        )
        phase_gauge = Gauge(
            mn.TRAINING_PHASE,
            "Current training phase (0=idle, 1=training, 2=checkpoint_saving)",
            labelnames=["rank", "node_id"],
            registry=self._registry,
        )

        self._iteration_child = iteration_gauge.labels(**self._labels)
        self._phase_child = phase_gauge.labels(**self._labels)
        self._iteration_child.set(0)
        self._phase_child.set(_PHASE_TO_NUMERIC["idle"])

        self._controller_handle: Any | None = None
        self._controller_lookup_failed: bool = False

        httpd, _thread = start_http_server(port=0, registry=self._registry)
        self._httpd = httpd
        self._port: int = httpd.server_port

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
        return f"http://localhost:{self._port}"

    def step(
        self,
        iteration: int,
        loss: float | None = None,
        grad_norm: float | None = None,
        mfu: float | None = None,
        iteration_time: float | None = None,
        phase: Literal["idle", "training", "checkpoint_saving"] = "training",
    ) -> None:
        try:
            self._step_inner(
                iteration=iteration,
                loss=loss,
                grad_norm=grad_norm,
                mfu=mfu,
                iteration_time=iteration_time,
                phase=phase,
            )
        except Exception:
            logger.warning(
                "FtMegatronAgent.step() failed at iteration=%d", iteration,
                exc_info=True,
            )

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    # ------------------------------------------------------------------
    # Internal: step logic
    # ------------------------------------------------------------------

    def _step_inner(
        self,
        iteration: int,
        loss: float | None,
        grad_norm: float | None,
        mfu: float | None,
        iteration_time: float | None,
        phase: Literal["idle", "training", "checkpoint_saving"],
    ) -> None:
        self._iteration_child.set(iteration)
        self._phase_child.set(_PHASE_TO_NUMERIC.get(phase, 0.0))

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
                    )

    def _get_controller_handle(self) -> Any | None:
        if self._controller_handle is not None:
            return self._controller_handle
        if self._controller_lookup_failed:
            return None

        try:
            import ray

            self._controller_handle = ray.get_actor("ft_controller")
        except Exception:
            self._controller_lookup_failed = True
            logger.warning("Failed to get ft_controller actor handle")
            return None

        return self._controller_handle

    def _reset_controller_handle(self) -> None:
        self._controller_handle = None
        self._controller_lookup_failed = False
