from __future__ import annotations

import logging
from http.server import HTTPServer

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

import miles.utils.ft.controller.metrics.metric_names as metric_names
from miles.utils.ft.adapters.types import JobStatus

logger = logging.getLogger(__name__)

_JOB_STATUS_TO_NUMERIC: dict[JobStatus, int] = {
    JobStatus.RUNNING: 1,
    JobStatus.STOPPED: 0,
    JobStatus.FAILED: -1,
    JobStatus.PENDING: 2,
}


class ControllerExporter:
    """Exposes Controller operational metrics via a dedicated Prometheus HTTP endpoint.

    Uses an isolated CollectorRegistry to avoid polluting the global REGISTRY
    shared by NodeAgent / FtTrainingRankAgent exporters.
    """

    def __init__(
        self,
        port: int = 9400,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self._port = port
        self._registry = registry or CollectorRegistry()
        self._httpd: HTTPServer | None = None
        self._mode = Gauge(
            metric_names.CONTROLLER_MODE,
            "Controller mode (0=monitoring, 1=recovery)",
            labelnames=["subsystem"],
            registry=self._registry,
        )
        self._mode.labels(subsystem="training").set(0)

        self._tick_count = Counter(
            metric_names.CONTROLLER_TICK_COUNT,
            "Cumulative tick count",
            registry=self._registry,
        )
        self._recovery_phase = Gauge(
            metric_names.CONTROLLER_RECOVERY_PHASE,
            "Recovery phase encoding (0=none, 1=check_alerts, 2=reattempting, ...)",
            labelnames=["subsystem"],
            registry=self._registry,
        )
        self._recovery_phase.labels(subsystem="training").set(0)

        self._main_job_status = Gauge(
            metric_names.MAIN_JOB_STATUS,
            "Training job status (-1=FAILED, 0=STOPPED, 1=RUNNING, 2=PENDING)",
            registry=self._registry,
        )
        self._training_loss_latest = Gauge(
            metric_names.TRAINING_LOSS_LATEST,
            "Latest training loss from rank 0",
            registry=self._registry,
        )
        self._training_mfu_latest = Gauge(
            metric_names.TRAINING_MFU_LATEST,
            "Latest training MFU from rank 0",
            registry=self._registry,
        )
        self._tick_duration_seconds = Histogram(
            metric_names.CONTROLLER_TICK_DURATION_SECONDS,
            "Wall-clock duration of each controller tick",
            registry=self._registry,
        )
        self._decision_total = Counter(
            metric_names.CONTROLLER_DECISION_TOTAL,
            "Total non-NONE decisions by action and trigger",
            labelnames=["action", "trigger"],
            registry=self._registry,
        )
        self._recovery_duration_seconds = Histogram(
            metric_names.CONTROLLER_RECOVERY_DURATION_SECONDS,
            "Duration of complete recovery cycles",
            registry=self._registry,
        )
        self._last_tick_timestamp = Gauge(
            metric_names.CONTROLLER_LAST_TICK_TIMESTAMP,
            "Wall-clock epoch timestamp of last completed tick",
            registry=self._registry,
        )

    @property
    def address(self) -> str:
        return f"http://localhost:{self._port}"

    def start(self) -> None:
        self._httpd, _thread = start_http_server(port=self._port, registry=self._registry)
        self._port = self._httpd.server_port
        logger.info("controller_exporter_started port=%d", self._port)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            logger.info("controller_exporter_stopped")

    def update_mode(self, *, is_recovery: bool, subsystem: str = "training") -> None:
        self._mode.labels(subsystem=subsystem).set(1 if is_recovery else 0)

    def update_tick_count(self) -> None:
        self._tick_count.inc()

    def update_recovery_phase(self, phase_int: int, *, subsystem: str = "training") -> None:
        self._recovery_phase.labels(subsystem=subsystem).set(phase_int)

    def update_main_job_status(self, status: JobStatus) -> None:
        self._main_job_status.set(_JOB_STATUS_TO_NUMERIC[status])

    def update_tick_duration(self, seconds: float) -> None:
        self._tick_duration_seconds.observe(seconds)

    def record_decision(self, action: str, trigger: str) -> None:
        self._decision_total.labels(action=action, trigger=trigger).inc()

    def observe_recovery_duration(self, seconds: float) -> None:
        self._recovery_duration_seconds.observe(seconds)

    def update_last_tick_timestamp(self, timestamp: float) -> None:
        self._last_tick_timestamp.set(timestamp)

    def update_training_metrics(
        self,
        loss: float | None,
        mfu: float | None,
    ) -> None:
        if loss is not None:
            self._training_loss_latest.set(loss)
        if mfu is not None:
            self._training_mfu_latest.set(mfu)

    def update_subsystem_state(
        self,
        *,
        subsystem: str,
        is_recovery: bool,
        recovery_phase_int: int,
    ) -> None:
        self.update_mode(is_recovery=is_recovery, subsystem=subsystem)
        self.update_recovery_phase(recovery_phase_int, subsystem=subsystem)

    def update_from_state(
        self,
        *,
        job_status: JobStatus,
        subsystem_modes: dict[str, tuple[bool, int]],
        latest_loss: float | None,
        latest_mfu: float | None,
    ) -> None:
        self.update_main_job_status(job_status)
        self.update_tick_count()

        for name, (is_recovery, phase_int) in subsystem_modes.items():
            self.update_subsystem_state(
                subsystem=name,
                is_recovery=is_recovery,
                recovery_phase_int=phase_int,
            )
        self.update_training_metrics(loss=latest_loss, mfu=latest_mfu)


class NullControllerExporter(ControllerExporter):
    # Intentionally skips super().__init__() to avoid creating Prometheus metrics.
    # All public methods are overridden as no-ops. If ControllerExporter gains new
    # methods, they must be overridden here as well.
    """No-op exporter that silently discards all metrics.

    Used when no Prometheus endpoint is configured, eliminating
    ``if exporter is not None`` guards in the controller code.
    """

    def __init__(self) -> None:
        pass

    @property
    def address(self) -> str:
        return ""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def update_mode(self, *, is_recovery: bool, subsystem: str = "training") -> None:
        pass

    def update_tick_count(self) -> None:
        pass

    def update_recovery_phase(self, phase_int: int, *, subsystem: str = "training") -> None:
        pass

    def update_main_job_status(self, status: JobStatus) -> None:
        pass

    def update_tick_duration(self, seconds: float) -> None:
        pass

    def record_decision(self, action: str, trigger: str) -> None:
        pass

    def observe_recovery_duration(self, seconds: float) -> None:
        pass

    def update_last_tick_timestamp(self, timestamp: float) -> None:
        pass

    def update_training_metrics(self, loss: float | None, mfu: float | None) -> None:
        pass

    def update_subsystem_state(
        self,
        *,
        subsystem: str,
        is_recovery: bool,
        recovery_phase_int: int,
    ) -> None:
        pass

    def update_from_state(
        self,
        *,
        job_status: JobStatus,
        subsystem_modes: dict[str, tuple[bool, int]],
        latest_loss: float | None,
        latest_mfu: float | None,
    ) -> None:
        pass
