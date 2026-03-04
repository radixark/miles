from __future__ import annotations

import logging
import os
import socket

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

        httpd, _thread = start_http_server(port=0, registry=self._registry)
        self._httpd = httpd
        self._port: int = httpd.server_port

    def get_exporter_address(self) -> str:
        return f"http://localhost:{self._port}"
