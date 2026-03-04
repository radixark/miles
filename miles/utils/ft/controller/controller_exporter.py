from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

logger = logging.getLogger(__name__)


class ControllerExporter:
    """Exposes Controller operational metrics via a dedicated Prometheus HTTP endpoint.

    Uses an isolated CollectorRegistry to avoid polluting the global REGISTRY
    shared by NodeAgent / MegatronAgent exporters.
    """

    def __init__(
        self,
        port: int = 9400,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self._port = port
        self._registry = registry or CollectorRegistry()

        self._mode = Gauge(
            "ft_controller_mode",
            "Controller mode (0=monitoring, 1=recovery)",
            registry=self._registry,
        )
        self._tick_count = Counter(
            "ft_controller_tick_count",
            "Cumulative tick count",
            registry=self._registry,
        )
        self._evicted_node_count = Gauge(
            "ft_controller_evicted_node_count",
            "Number of currently evicted nodes",
            registry=self._registry,
        )
        self._recovery_phase = Gauge(
            "ft_controller_recovery_phase",
            "Recovery phase encoding (0=none, 1=check_alerts, 2=reattempting, ...)",
            registry=self._registry,
        )

        self._training_job_status = Gauge(
            "ft_training_job_status",
            "Training job status (-1=FAILED, 0=STOPPED, 1=RUNNING, 2=PENDING)",
            registry=self._registry,
        )
        self._training_loss_latest = Gauge(
            "ft_training_loss_latest",
            "Latest training loss (mean across ranks)",
            registry=self._registry,
        )
        self._training_mfu_latest = Gauge(
            "ft_training_mfu_latest",
            "Latest training MFU (mean across ranks)",
            registry=self._registry,
        )

    @property
    def address(self) -> str:
        return f"http://localhost:{self._port}"

    def start(self) -> None:
        start_http_server(port=self._port, registry=self._registry)
        logger.info("controller_exporter_started port=%d", self._port)

    def update_mode(self, mode: int) -> None:
        self._mode.set(mode)

    def update_tick_count(self) -> None:
        self._tick_count.inc()

    def update_evicted_node_count(self, count: int) -> None:
        self._evicted_node_count.set(count)

    def update_recovery_phase(self, phase: int) -> None:
        self._recovery_phase.set(phase)

    def update_training_job_status(self, status: int) -> None:
        self._training_job_status.set(status)

    def update_training_metrics(
        self,
        loss: float | None,
        mfu: float | None,
    ) -> None:
        if loss is not None:
            self._training_loss_latest.set(loss)
        if mfu is not None:
            self._training_mfu_latest.set(mfu)
