from __future__ import annotations

import logging
import math
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from miles.utils.ft.controller.controller_exporter import ControllerExporter
from miles.utils.ft.controller.detectors._metric_names import (
    NODE_DISK_AVAILABLE_BYTES,
    NODE_GPU_AVAILABLE,
    NODE_NIC_UP,
    NODE_XID_CODE_RECENT,
)
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import (
    ActionType,
    Decision,
    RecoveryPhase,
    RECOVERY_PHASE_TO_INT,
)
from miles.utils.ft.platform.protocols import (
    DiagnosticSchedulerProtocol,
    JobStatus,
    NodeManagerProtocol,
    NotificationProtocol,
    TrainingJobProtocol,
)

logger = logging.getLogger(__name__)

_CRITICAL_XID_CODES: frozenset[int] = frozenset({48, 62, 64, 79})
_DISK_AVAILABLE_THRESHOLD_BYTES: float = 1e9
_PENDING_TIMEOUT_SECONDS: int = 300
_MAX_RETRIES: int = 3

_StepHandler = Callable[[], Coroutine[Any, Any, None]]


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


@dataclass
class RecoveryContext:
    trigger: str
    phase: RecoveryPhase = RecoveryPhase.CHECK_ALERTS
    reattempt_start_time: datetime | None = None
    reattempt_base_iteration: int | None = None
    recovery_start_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    global_timeout_seconds: int = 1800
    monitoring_success_iterations: int = 10
    monitoring_timeout_seconds: int = 600
    phase_before_notify: RecoveryPhase | None = None


class RecoveryOrchestrator:
    def __init__(
        self,
        trigger: str,
        node_manager: NodeManagerProtocol,
        training_job: TrainingJobProtocol,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        notifier: NotificationProtocol | None,
        diagnostic_scheduler: DiagnosticSchedulerProtocol,
        controller_exporter: ControllerExporter | None = None,
        global_timeout_seconds: int = 1800,
        monitoring_success_iterations: int = 10,
        monitoring_timeout_seconds: int = 600,
    ) -> None:
        self._node_manager = node_manager
        self._training_job = training_job
        self._metric_store = metric_store
        self._mini_wandb = mini_wandb
        self._notifier = notifier
        self._diagnostic_scheduler = diagnostic_scheduler
        self._controller_exporter = controller_exporter

        self._context = RecoveryContext(
            trigger=trigger,
            global_timeout_seconds=global_timeout_seconds,
            monitoring_success_iterations=monitoring_success_iterations,
            monitoring_timeout_seconds=monitoring_timeout_seconds,
        )

        self._reattempt_submitted: bool = False
        self._reattempt_submit_time: datetime | None = None
        self._bad_node_ids: list[str] = []

    @property
    def phase(self) -> RecoveryPhase:
        return self._context.phase

    @property
    def trigger(self) -> str:
        return self._context.trigger

    def is_done(self) -> bool:
        return self._context.phase == RecoveryPhase.DONE

    async def step(self) -> None:
        if self.is_done():
            return

        if self._check_global_timeout():
            return

        phase_handlers: dict[RecoveryPhase, _StepHandler] = {
            RecoveryPhase.CHECK_ALERTS: self._step_check_alerts,
            RecoveryPhase.REATTEMPTING: self._step_reattempting,
            RecoveryPhase.MONITORING: self._step_monitoring,
            RecoveryPhase.DIAGNOSING: self._step_diagnosing,
            RecoveryPhase.EVICT_AND_RESTART: self._step_evict_and_restart,
            RecoveryPhase.NOTIFY: self._step_notify,
        }

        handler = phase_handlers.get(self._context.phase)
        if handler is not None:
            await handler()

        self._update_exporter()

    # -------------------------------------------------------------------
    # Global timeout
    # -------------------------------------------------------------------

    def _check_global_timeout(self) -> bool:
        if self._context.phase in (RecoveryPhase.NOTIFY, RecoveryPhase.DONE):
            return False

        elapsed = (
            datetime.now(timezone.utc) - self._context.recovery_start_time
        ).total_seconds()
        if elapsed > self._context.global_timeout_seconds:
            logger.warning(
                "recovery_global_timeout elapsed=%.0f phase=%s trigger=%s",
                elapsed, self._context.phase.value, self._context.trigger,
            )
            self._transition(RecoveryPhase.NOTIFY)
            return True
        return False

    # -------------------------------------------------------------------
    # Phase handlers
    # -------------------------------------------------------------------

    async def _step_check_alerts(self) -> None:
        bad_nodes: set[str] = set()
        reasons: list[str] = []

        self._query_gpu_lost(bad_nodes, reasons)
        self._query_critical_xid(bad_nodes, reasons)
        self._query_disk_fault(bad_nodes, reasons)
        self._query_majority_nic_down(bad_nodes, reasons)

        if bad_nodes:
            self._bad_node_ids = sorted(bad_nodes)
            logger.info(
                "check_alerts_found bad_nodes=%s reasons=%s",
                self._bad_node_ids, reasons,
            )
            self._transition(RecoveryPhase.EVICT_AND_RESTART)
        else:
            logger.info("check_alerts_clean trigger=%s", self._context.trigger)
            self._transition(RecoveryPhase.REATTEMPTING)

    async def _step_reattempting(self) -> None:
        if not self._reattempt_submitted:
            try:
                await self._training_job.stop_training()
            except Exception:
                logger.warning("reattempt_stop_training_failed", exc_info=True)

            self._mini_wandb.clear()

            try:
                await self._training_job.submit_training()
            except Exception:
                logger.error("reattempt_submit_training_failed", exc_info=True)
                self._transition(RecoveryPhase.NOTIFY)
                return

            self._reattempt_submitted = True
            self._reattempt_submit_time = datetime.now(timezone.utc)
            logger.info("reattempt_submitted trigger=%s", self._context.trigger)
            return

        status = await self._training_job.get_training_status()

        if status == JobStatus.RUNNING:
            iteration = self._mini_wandb.latest(metric_name="iteration", rank=0)
            self._context.reattempt_start_time = datetime.now(timezone.utc)
            self._context.reattempt_base_iteration = (
                int(iteration) if iteration is not None and _is_finite(iteration) else 0
            )
            logger.info(
                "reattempt_running base_iteration=%s",
                self._context.reattempt_base_iteration,
            )
            self._transition(RecoveryPhase.MONITORING)
            return

        if status == JobStatus.FAILED:
            logger.warning("reattempt_immediately_failed trigger=%s", self._context.trigger)
            self._transition(RecoveryPhase.DIAGNOSING)
            return

        if self._reattempt_submit_time is not None:
            elapsed = (
                datetime.now(timezone.utc) - self._reattempt_submit_time
            ).total_seconds()
            if elapsed > _PENDING_TIMEOUT_SECONDS:
                logger.warning(
                    "reattempt_pending_timeout elapsed=%.0f", elapsed,
                )
                self._transition(RecoveryPhase.NOTIFY)

    async def _step_monitoring(self) -> None:
        status = await self._training_job.get_training_status()
        progress = self._iteration_progress()

        if status == JobStatus.FAILED:
            logger.warning(
                "monitoring_training_failed progress_iterations=%d trigger=%s",
                progress, self._context.trigger,
            )
            self._transition(RecoveryPhase.DIAGNOSING)
            return

        if status == JobStatus.RUNNING and progress >= self._context.monitoring_success_iterations:
            logger.info(
                "monitoring_success progress_iterations=%d threshold=%d",
                progress, self._context.monitoring_success_iterations,
            )
            self._transition(RecoveryPhase.DONE)
            return

        if self._context.reattempt_start_time is not None:
            elapsed = (
                datetime.now(timezone.utc) - self._context.reattempt_start_time
            ).total_seconds()
            if elapsed > self._context.monitoring_timeout_seconds:
                logger.warning(
                    "monitoring_timeout elapsed=%.0f trigger=%s",
                    elapsed, self._context.trigger,
                )
                self._transition(RecoveryPhase.DIAGNOSING)

    def _iteration_progress(self) -> int:
        current_iteration = self._mini_wandb.latest(metric_name="iteration", rank=0)
        if current_iteration is None or not _is_finite(current_iteration):
            return 0
        base = self._context.reattempt_base_iteration or 0
        return int(current_iteration) - base

    async def _step_diagnosing(self) -> None:
        decision = await self._diagnostic_scheduler.run_diagnostic_pipeline(
            trigger_reason=self._context.trigger,
        )

        if decision.action == ActionType.MARK_BAD_AND_RESTART:
            self._bad_node_ids = list(decision.bad_node_ids)
            logger.info(
                "diagnosing_found_bad_nodes bad_nodes=%s", self._bad_node_ids,
            )
            self._transition(RecoveryPhase.EVICT_AND_RESTART)
        else:
            logger.info("diagnosing_all_passed trigger=%s", self._context.trigger)
            self._transition(RecoveryPhase.NOTIFY)

    async def _step_evict_and_restart(self) -> None:
        for node_id in self._bad_node_ids:
            success = await self._retry_async(
                lambda nid=node_id: self._node_manager.mark_node_bad(
                    nid, reason=f"recovery eviction: {self._context.trigger}",
                ),
                description=f"mark_node_bad({node_id})",
            )
            if not success:
                self._transition(RecoveryPhase.NOTIFY)
                return

        try:
            await self._training_job.stop_training()
        except Exception:
            logger.warning("evict_stop_training_failed", exc_info=True)

        self._mini_wandb.clear()

        success = await self._retry_async(
            self._training_job.submit_training,
            description="submit_training",
        )
        if not success:
            self._transition(RecoveryPhase.NOTIFY)
            return

        logger.info(
            "evict_and_restart_done bad_nodes=%s trigger=%s",
            self._bad_node_ids, self._context.trigger,
        )
        self._transition(RecoveryPhase.DONE)

    async def _step_notify(self) -> None:
        prev = self._context.phase_before_notify
        message = (
            f"Recovery requires human intervention. "
            f"trigger={self._context.trigger} "
            f"phase_before_notify={prev.value if prev else 'unknown'}"
        )
        logger.warning("recovery_notify reason=%s", message)

        if self._notifier is not None:
            try:
                await self._notifier.send(
                    title="Recovery Alert",
                    content=message,
                    severity="critical",
                )
            except Exception:
                logger.exception("recovery_notifier_send_failed")

        self._transition(RecoveryPhase.DONE)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _transition(self, new_phase: RecoveryPhase) -> None:
        old = self._context.phase
        if new_phase == RecoveryPhase.NOTIFY:
            self._context.phase_before_notify = old
        self._context.phase = new_phase
        logger.info("recovery_transition %s -> %s", old.value, new_phase.value)
        self._update_exporter()

    def _update_exporter(self) -> None:
        if self._controller_exporter is None:
            return
        phase_int = RECOVERY_PHASE_TO_INT.get(self._context.phase, 0)
        self._controller_exporter.update_recovery_phase(phase_int)

    async def _retry_async(
        self,
        func: Callable[[], Coroutine[Any, Any, Any]],
        description: str,
        max_retries: int = _MAX_RETRIES,
    ) -> bool:
        for attempt in range(max_retries):
            try:
                await func()
                return True
            except Exception:
                logger.warning(
                    "retry_failed description=%s attempt=%d/%d",
                    description, attempt + 1, max_retries,
                    exc_info=True,
                )
        logger.error("retry_exhausted description=%s", description)
        return False

    # -------------------------------------------------------------------
    # CHECK_ALERTS queries (reuses hardware detector patterns)
    # -------------------------------------------------------------------

    def _query_gpu_lost(
        self, bad_nodes: set[str], reasons: list[str],
    ) -> None:
        df = self._metric_store.instant_query(f"{NODE_GPU_AVAILABLE} == 0")
        if df.is_empty():
            return
        for node_id in df["node_id"].unique().to_list():
            bad_nodes.add(node_id)
            reasons.append(f"GPU unavailable on {node_id}")

    def _query_critical_xid(
        self, bad_nodes: set[str], reasons: list[str],
    ) -> None:
        df = self._metric_store.instant_query(NODE_XID_CODE_RECENT)
        if df.is_empty():
            return
        for row in df.iter_rows(named=True):
            xid_code = int(row.get("xid", -1))
            if xid_code in _CRITICAL_XID_CODES:
                node_id = row["node_id"]
                bad_nodes.add(node_id)
                reasons.append(f"critical XID {xid_code} on {node_id}")

    def _query_disk_fault(
        self, bad_nodes: set[str], reasons: list[str],
    ) -> None:
        df = self._metric_store.instant_query(
            f"{NODE_DISK_AVAILABLE_BYTES} < {_DISK_AVAILABLE_THRESHOLD_BYTES}"
        )
        if df.is_empty():
            return
        for row in df.iter_rows(named=True):
            node_id = row["node_id"]
            bad_nodes.add(node_id)
            reasons.append(f"disk space low on {node_id}")

    def _query_majority_nic_down(
        self, bad_nodes: set[str], reasons: list[str],
    ) -> None:
        df = self._metric_store.instant_query(NODE_NIC_UP)
        if df.is_empty():
            return

        node_stats: dict[str, tuple[int, int]] = {}
        for row in df.iter_rows(named=True):
            node_id = row["node_id"]
            down_count, total_count = node_stats.get(node_id, (0, 0))
            total_count += 1
            if row["value"] == 0.0:
                down_count += 1
            node_stats[node_id] = (down_count, total_count)

        for node_id, (down_count, total_count) in node_stats.items():
            if total_count > 0 and down_count > total_count / 2:
                bad_nodes.add(node_id)
                reasons.append(f"majority NIC down on {node_id} ({down_count}/{total_count})")
