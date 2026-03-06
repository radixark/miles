from __future__ import annotations

import asyncio
import logging
import time

from miles.utils.ft.controller.actions import (
    PlatformDeps,
    handle_mark_bad_and_restart,
    handle_notify_human,
)
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.diagnostics.scheduler import DiagnosticScheduler
from miles.utils.ft.controller.metrics import start_metric_store_task, stop_metric_store_task
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.rank_registry import RankRegistry
from miles.utils.ft.controller.recovery_cooldown import RecoveryCooldown
from miles.utils.ft.controller.recovery_lifecycle import RecoveryLifecycleManager
from miles.utils.ft.models._fault import ActionType, Decision
from miles.utils.ft.models._recovery import (
    ControllerMode,
    ControllerStatus,
    _BAD_NODES_CONFIRMED_PHASES,
)
from miles.utils.ft.protocols.metrics import MetricStoreProtocol
from miles.utils.ft.protocols.platform import (
    DiagnosticSchedulerProtocol,
    JobStatus,
    NodeManagerProtocol,
    NotificationProtocol,
    TrainingJobProtocol,
)

logger = logging.getLogger(__name__)

_ALL_DETECTORS_PASSED = Decision(action=ActionType.NONE, reason="all detectors passed")


class FtController:
    def __init__(
        self,
        node_manager: NodeManagerProtocol,
        training_job: TrainingJobProtocol,
        metric_store: MetricStoreProtocol,
        rank_registry: RankRegistry,
        notifier: NotificationProtocol | None = None,
        detectors: list[BaseFaultDetector] | None = None,
        tick_interval: float = 30.0,
        controller_exporter: ControllerExporter | None = None,
        diagnostic_scheduler: DiagnosticSchedulerProtocol | None = None,
        recovery_cooldown: RecoveryCooldown | None = None,
        registration_grace_ticks: int = 5,
    ) -> None:
        self._training_job = training_job
        self._metric_store = metric_store
        self._rank_registry = rank_registry
        self._mini_wandb = rank_registry.mini_wandb
        self._detectors: list[BaseFaultDetector] = detectors or []
        self._tick_interval = tick_interval
        self._controller_exporter = controller_exporter
        self._registration_grace_ticks = registration_grace_ticks

        resolved_scheduler: DiagnosticSchedulerProtocol = (
            diagnostic_scheduler
            or DiagnosticScheduler(
                agents=self._rank_registry.agents,
                pipeline=["gpu"],
                rank_pids_provider=self._rank_registry.get_rank_pids_for_node,
            )
        )
        self._platform_deps = PlatformDeps(
            node_manager=node_manager,
            training_job=training_job,
            metric_store=metric_store,
            mini_wandb=self._mini_wandb,
            notifier=notifier,
            diagnostic_scheduler=resolved_scheduler,
            controller_exporter=controller_exporter,
        )

        duration_cb = (
            controller_exporter.observe_recovery_duration
            if controller_exporter is not None
            else None
        )
        self._recovery_manager = RecoveryLifecycleManager(
            cooldown=recovery_cooldown or RecoveryCooldown(window_minutes=30.0, max_count=3),
            on_recovery_duration=duration_cb,
        )

        self._shutting_down: bool = False
        self._tick_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def rank_registry(self) -> RankRegistry:
        return self._rank_registry

    @property
    def recovery_manager(self) -> RecoveryLifecycleManager:
        return self._recovery_manager

    async def submit_initial_training(self) -> str:
        return await self._training_job.submit_training()

    async def run(self) -> None:
        logger.info("controller_start tick_interval=%s", self._tick_interval)
        scrape_task = await start_metric_store_task(self._metric_store)
        try:
            while not self._shutting_down:
                await self._tick()
                if not self._shutting_down:
                    await asyncio.sleep(self._tick_interval)
        finally:
            await self._stop_services(scrape_task)
        logger.info("controller_stopped")

    async def shutdown(self) -> None:
        logger.info("controller_shutdown_requested")
        self._shutting_down = True

    def get_status(self) -> ControllerStatus:
        rm = self._recovery_manager

        if rm.in_progress:
            mode = ControllerMode.RECOVERY
            recovery_phase = rm.phase
            phase_history: list | None = list(rm.orchestrator.phase_history) if rm.orchestrator else None
            bad_nodes_confirmed = recovery_phase in _BAD_NODES_CONFIRMED_PHASES
        else:
            mode = ControllerMode.MONITORING
            recovery_phase = None
            phase_history = rm.last_phase_history
            bad_nodes_confirmed = False

        iteration_val = self._mini_wandb.latest(metric_name="iteration")
        latest_iteration = int(iteration_val) if iteration_val is not None else None

        return ControllerStatus(
            mode=mode,
            recovery_phase=recovery_phase,
            phase_history=phase_history,
            tick_count=self._tick_count,
            active_run_id=self._rank_registry.active_run_id,
            bad_nodes=sorted(rm.diagnosing_nodes),
            recovery_in_progress=rm.in_progress,
            bad_nodes_confirmed=bad_nodes_confirmed,
            latest_iteration=latest_iteration,
        )

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        self._tick_count += 1
        t0 = time.monotonic()
        job_status: JobStatus | None = None
        try:
            job_status = await self._tick_inner()
        except Exception:
            logger.error("tick_failed tick=%d", self._tick_count, exc_info=True)
        finally:
            duration = time.monotonic() - t0
            if self._controller_exporter is not None:
                self._controller_exporter.update_tick_duration(duration)
                self._controller_exporter.update_last_tick_timestamp(time.time())
            if job_status is not None:
                self._update_exporter_metrics(job_status)

    async def _tick_inner(self) -> JobStatus:
        self._rank_registry.warn_if_incomplete()
        job_status = await self._training_job.get_training_status()

        if self._recovery_manager.in_progress:
            self._run_critical_detectors_during_recovery(job_status)
            await self._recovery_manager.step()
            return job_status

        if not self._should_run_detectors():
            return job_status

        ctx = self._build_detector_context(job_status)
        decision = self._evaluate_detectors(ctx)

        logger.info(
            "loop_tick tick=%d active_run_id=%s decision_action=%s decision_reason=%s",
            self._tick_count, self._rank_registry.active_run_id,
            decision.action.value, decision.reason,
        )

        await self._execute_decision(decision)
        return job_status

    def _should_run_detectors(self) -> bool:
        if len(self._rank_registry.rank_placement) == 0:
            logger.info("skip_detectors_no_ranks tick=%d", self._tick_count)
            return False

        if self._tick_count <= self._registration_grace_ticks:
            logger.info(
                "skip_detectors_grace_period tick=%d grace_ticks=%d",
                self._tick_count, self._registration_grace_ticks,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------

    def _build_detector_context(self, job_status: JobStatus) -> DetectorContext:
        return DetectorContext(
            metric_store=self._metric_store,
            mini_wandb=self._mini_wandb,
            rank_placement=dict(self._rank_registry.rank_placement),
            job_status=job_status,
        )

    def _evaluate_detectors(self, ctx: DetectorContext) -> Decision:
        for detector in self._detectors:
            try:
                decision = detector.evaluate(ctx)
            except Exception:
                logger.error(
                    "detector_evaluate_failed detector=%s",
                    type(detector).__name__,
                    exc_info=True,
                )
                continue
            if decision.action != ActionType.NONE:
                return decision

        return _ALL_DETECTORS_PASSED

    def _run_critical_detectors_during_recovery(self, job_status: JobStatus) -> None:
        """Run is_critical detectors even while recovery is active.

        If a critical detector fires MARK_BAD_AND_RESTART with new bad nodes,
        merge them into the orchestrator's eviction list.
        """
        ctx = self._build_detector_context(job_status)

        for detector in self._detectors:
            if not detector.is_critical:
                continue
            try:
                decision = detector.evaluate(ctx)
            except Exception:
                logger.error(
                    "critical_detector_failed detector=%s",
                    type(detector).__name__,
                    exc_info=True,
                )
                continue

            if decision.action == ActionType.MARK_BAD_AND_RESTART and decision.bad_node_ids:
                self._recovery_manager.add_bad_nodes(decision.bad_node_ids)

    # ------------------------------------------------------------------
    # Decision execution
    # ------------------------------------------------------------------

    async def _execute_decision(self, decision: Decision) -> None:
        if decision.action == ActionType.NONE:
            return

        trigger_str = decision.trigger.value if decision.trigger else "unknown"
        logger.info(
            "decision_event decision_action=%s trigger=%s bad_node_ids=%s run_id=%s tick=%d",
            decision.action.value, trigger_str, decision.bad_node_ids,
            self._rank_registry.active_run_id, self._tick_count,
        )
        if self._controller_exporter is not None:
            self._controller_exporter.record_decision(
                action=decision.action.value, trigger=trigger_str,
            )

        if decision.action == ActionType.MARK_BAD_AND_RESTART:
            await handle_mark_bad_and_restart(decision=decision, deps=self._platform_deps)

        elif decision.action == ActionType.ENTER_RECOVERY:
            started = await self._recovery_manager.start(
                decision=decision, deps=self._platform_deps,
            )
            if not started:
                await handle_notify_human(
                    decision=Decision(
                        action=ActionType.NOTIFY_HUMAN,
                        reason=f"Recovery cooldown: {decision.trigger.value} triggered too many times",
                        trigger=decision.trigger,
                    ),
                    notifier=self._platform_deps.notifier,
                )

        elif decision.action == ActionType.NOTIFY_HUMAN:
            await handle_notify_human(
                decision=decision, notifier=self._platform_deps.notifier,
            )

        else:
            raise ValueError(f"Unknown action type: {decision.action}")

    # ------------------------------------------------------------------
    # Exporter metrics
    # ------------------------------------------------------------------

    def _update_exporter_metrics(self, job_status: JobStatus) -> None:
        if self._controller_exporter is None:
            return

        is_recovery = self._recovery_manager.in_progress
        self._controller_exporter.update_from_state(
            job_status=job_status,
            mode=ControllerMode.RECOVERY if is_recovery else ControllerMode.MONITORING,
            recovery_phase=self._recovery_manager.phase if is_recovery else None,
            latest_loss=self._mini_wandb.latest(metric_name="loss"),
            latest_mfu=self._mini_wandb.latest(metric_name="mfu"),
        )

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    async def _stop_services(self, scrape_task: asyncio.Task[None] | None) -> None:
        await stop_metric_store_task(self._metric_store, scrape_task)
        if self._controller_exporter is not None:
            self._controller_exporter.stop()
        if self._platform_deps.notifier is not None:
            try:
                await self._platform_deps.notifier.aclose()
            except Exception:
                logger.warning("notifier_aclose_failed", exc_info=True)
