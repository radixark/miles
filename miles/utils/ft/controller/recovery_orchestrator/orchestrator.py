from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.metrics.protocol import MetricStoreProtocol
from miles.utils.ft.controller.recovery_orchestrator.alert_checker import AlertChecker
from miles.utils.ft.controller.recovery_orchestrator.context import RecoveryContext
from miles.utils.ft.controller.recovery_orchestrator.phase_handlers import (
    step_check_alerts,
    step_diagnosing,
    step_evict_and_restart,
    step_monitoring,
    step_notify,
    step_reattempting,
)
from miles.utils.ft.models import RecoveryPhase, TriggerType
from miles.utils.ft.platform.protocols import (
    DiagnosticSchedulerProtocol,
    NodeManagerProtocol,
    NotificationProtocol,
    TrainingJobProtocol,
)

logger = logging.getLogger(__name__)


class RecoveryOrchestrator:
    def __init__(
        self,
        trigger: TriggerType,
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
        self._mini_wandb = mini_wandb
        self._notifier = notifier
        self._diagnostic_scheduler = diagnostic_scheduler
        self._controller_exporter = controller_exporter
        self._alert_checker = AlertChecker(metric_store=metric_store)

        self._context = RecoveryContext(
            trigger=trigger,
            global_timeout_seconds=global_timeout_seconds,
            monitoring_success_iterations=monitoring_success_iterations,
            monitoring_timeout_seconds=monitoring_timeout_seconds,
        )

    @property
    def phase(self) -> RecoveryPhase:
        return self._context.phase

    @property
    def trigger(self) -> TriggerType:
        return self._context.trigger

    @property
    def bad_node_ids(self) -> list[str]:
        return self._context.bad_node_ids

    def is_done(self) -> bool:
        return self._context.phase == RecoveryPhase.DONE

    async def step(self) -> None:
        if self.is_done():
            return

        if self._check_global_timeout():
            return

        next_phase = await self._dispatch_phase()
        if next_phase is not None:
            self._transition(next_phase)

        self._update_exporter()

    async def _dispatch_phase(self) -> RecoveryPhase | None:
        ctx = self._context
        handlers: dict[RecoveryPhase, Callable[[], Awaitable[RecoveryPhase | None]]] = {
            RecoveryPhase.CHECK_ALERTS: lambda: step_check_alerts(ctx, self._alert_checker),
            RecoveryPhase.REATTEMPTING: lambda: step_reattempting(ctx, self._training_job, self._mini_wandb),
            RecoveryPhase.MONITORING: lambda: step_monitoring(ctx, self._training_job, self._mini_wandb),
            RecoveryPhase.DIAGNOSING: lambda: step_diagnosing(ctx, self._diagnostic_scheduler),
            RecoveryPhase.EVICT_AND_RESTART: lambda: step_evict_and_restart(
                ctx, self._node_manager, self._training_job, self._mini_wandb,
            ),
            RecoveryPhase.NOTIFY: lambda: step_notify(ctx, self._notifier),
        }

        handler = handlers.get(ctx.phase)
        if handler is None:
            return None
        return await handler()

    def _check_global_timeout(self) -> bool:
        if self._context.phase in (RecoveryPhase.NOTIFY, RecoveryPhase.DONE):
            return False

        elapsed = (datetime.now(timezone.utc) - self._context.recovery_start_time).total_seconds()
        if elapsed > self._context.global_timeout_seconds:
            logger.warning(
                "recovery_global_timeout elapsed=%.0f phase=%s trigger=%s",
                elapsed,
                self._context.phase.value,
                self._context.trigger,
            )
            self._transition(RecoveryPhase.NOTIFY)
            return True
        return False

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
        self._controller_exporter.update_recovery_phase(self._context.phase)
