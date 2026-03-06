from __future__ import annotations

import logging
from http.server import HTTPServer

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

import miles.utils.ft.metric_names as mn
from miles.utils.ft.models.recovery import RECOVERY_PHASE_TO_INT, ControllerMode, RecoveryPhase
from miles.utils.ft.protocols.platform import JobStatus

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
            mn.CONTROLLER_MODE,
            "Controller mode (0=monitoring, 1=recovery)",
            registry=self._registry,
        )
        self._tick_count = Counter(
            mn.CONTROLLER_TICK_COUNT,
            "Cumulative tick count",
            registry=self._registry,
        )
        self._recovery_phase = Gauge(
            mn.CONTROLLER_RECOVERY_PHASE,
            "Recovery phase encoding (0=none, 1=check_alerts, 2=reattempting, ...)",
            registry=self._registry,
        )

        self._training_job_status = Gauge(
            mn.TRAINING_JOB_STATUS,
            "Training job status (-1=FAILED, 0=STOPPED, 1=RUNNING, 2=PENDING)",
            registry=self._registry,
        )
        self._training_loss_latest = Gauge(
            mn.TRAINING_LOSS_LATEST,
            "Latest training loss from rank 0",
            registry=self._registry,
        )
        self._training_mfu_latest = Gauge(
            mn.TRAINING_MFU_LATEST,
            "Latest training MFU from rank 0",
            registry=self._registry,
        )
        self._tick_duration_seconds = Histogram(
            mn.CONTROLLER_TICK_DURATION_SECONDS,
            "Wall-clock duration of each controller tick",
            registry=self._registry,
        )
        self._decision_total = Counter(
            mn.CONTROLLER_DECISION_TOTAL,
            "Total non-NONE decisions by action and trigger",
            labelnames=["action", "trigger"],
            registry=self._registry,
        )
        self._recovery_duration_seconds = Histogram(
            mn.CONTROLLER_RECOVERY_DURATION_SECONDS,
            "Duration of complete recovery cycles",
            registry=self._registry,
        )
        self._last_tick_timestamp = Gauge(
            mn.CONTROLLER_LAST_TICK_TIMESTAMP,
            "Wall-clock epoch timestamp of last completed tick",
            registry=self._registry,
        )

    @property
    def address(self) -> str:
        return f"http://localhost:{self._port}"

    def start(self) -> None:
        self._httpd, _thread = start_http_server(port=self._port, registry=self._registry)
        logger.info("controller_exporter_started port=%d", self._port)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            logger.info("controller_exporter_stopped")

    def update_mode(self, *, is_recovery: bool) -> None:
        self._mode.set(1 if is_recovery else 0)

    def update_tick_count(self) -> None:
        self._tick_count.inc()

    def update_recovery_phase(self, phase: RecoveryPhase | None) -> None:
        self._recovery_phase.set(RECOVERY_PHASE_TO_INT.get(phase, 0) if phase else 0)

    def update_training_job_status(self, status: JobStatus) -> None:
        self._training_job_status.set(_JOB_STATUS_TO_NUMERIC.get(status, 0))

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

    def update_from_state(
        self,
        *,
        job_status: JobStatus,
        mode: ControllerMode,
        recovery_phase: RecoveryPhase | None,
        latest_loss: float | None,
        latest_mfu: float | None,
    ) -> None:
        self.update_training_job_status(job_status)
        self.update_tick_count()
        self.update_mode(is_recovery=(mode == ControllerMode.RECOVERY))
        if mode != ControllerMode.RECOVERY:
            self.update_recovery_phase(None)
        self.update_training_metrics(loss=latest_loss, mfu=latest_mfu)
