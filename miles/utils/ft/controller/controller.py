from __future__ import annotations

import asyncio
import logging
from typing import Any

from miles.utils.ft.controller.actions import (
    handle_enter_recovery,
    handle_mark_bad_and_restart,
    handle_notify_human,
)
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.diagnostics.scheduler import DiagnosticScheduler
from miles.utils.ft.controller.metrics import start_metric_store_task, stop_metric_store_task
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.protocol import MetricStoreProtocol
from miles.utils.ft.controller.rank_registry import RankRegistry
from miles.utils.ft.controller.recovery_orchestrator import RecoveryOrchestrator
from miles.utils.ft.models import ActionType, Decision, NodeAgentProtocol
from miles.utils.ft.platform.protocols import (
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
    ) -> None:
        self._node_manager = node_manager
        self._training_job = training_job
        self._metric_store = metric_store
        self._rank_registry = rank_registry
        self._mini_wandb = rank_registry.mini_wandb
        self._notifier = notifier
        self._detectors: list[BaseFaultDetector] = detectors or []
        self._tick_interval = tick_interval
        self._controller_exporter = controller_exporter

        self._diagnostic_scheduler: DiagnosticSchedulerProtocol = (
            diagnostic_scheduler
            or DiagnosticScheduler(
                agents=self._rank_registry.agents,
                pipeline=["gpu"],
                rank_pids_provider=self._rank_registry.get_rank_pids_for_node,
            )
        )

        self._shutting_down: bool = False
        self._tick_count: int = 0
        self._recovery_orchestrator: RecoveryOrchestrator | None = None
        self._diagnosing_nodes: set[str] = set()

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

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
            await stop_metric_store_task(self._metric_store, scrape_task)
            if self._controller_exporter is not None:
                self._controller_exporter.stop()
            if self._notifier is not None:
                try:
                    await self._notifier.aclose()
                except Exception:
                    logger.warning("notifier_aclose_failed", exc_info=True)
        logger.info("controller_stopped")

    async def shutdown(self) -> None:
        logger.info("controller_shutdown_requested")
        self._shutting_down = True

    def get_status(self) -> dict[str, Any]:
        recovery_phase: str | None = None
        if self._recovery_orchestrator is not None:
            mode = "recovery"
            recovery_phase = self._recovery_orchestrator.phase.value
        else:
            mode = "monitoring"

        bad_nodes: list[str] = sorted(self._diagnosing_nodes)
        return {
            "mode": mode,
            "recovery_phase": recovery_phase,
            "tick_count": self._tick_count,
            "active_run_id": self._rank_registry.active_run_id,
            "bad_nodes": bad_nodes,
        }

    # -------------------------------------------------------------------
    # Agent management (delegated to RankRegistry)
    # -------------------------------------------------------------------

    def register_agent(self, node_id: str, agent: NodeAgentProtocol) -> None:
        self._rank_registry.register_agent(node_id=node_id, agent=agent)

    async def log_step(
        self,
        run_id: str,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        self._rank_registry.log_step(
            run_id=run_id, step=step, metrics=metrics,
        )

    async def register_rank(
        self,
        run_id: str,
        rank: int,
        world_size: int,
        node_id: str,
        exporter_address: str,
        pid: int | None = None,
    ) -> None:
        self._rank_registry.register_rank(
            run_id=run_id,
            rank=rank,
            world_size=world_size,
            node_id=node_id,
            exporter_address=exporter_address,
            pid=pid,
        )

    # -------------------------------------------------------------------
    # Main loop tick
    # -------------------------------------------------------------------

    async def _tick(self) -> None:
        self._tick_count += 1

        if (
            self._rank_registry.expected_world_size is not None
            and len(self._rank_registry.rank_placement) < self._rank_registry.expected_world_size
        ):
            logger.warning(
                "incomplete_rank_registration registered=%d expected=%d run_id=%s",
                len(self._rank_registry.rank_placement),
                self._rank_registry.expected_world_size,
                self._rank_registry.active_run_id,
            )

        job_status = await self._training_job.get_training_status()

        if self._recovery_orchestrator is not None:
            await self._recovery_orchestrator.step()
            self._diagnosing_nodes = set(self._recovery_orchestrator.bad_node_ids)
            if self._recovery_orchestrator.is_done():
                logger.info("recovery_complete trigger=%s", self._recovery_orchestrator.trigger)
                self._recovery_orchestrator = None
                self._diagnosing_nodes.clear()
            self._update_exporter_metrics(job_status)
            return

        ctx = DetectorContext(
            metric_store=self._metric_store,
            mini_wandb=self._mini_wandb,
            rank_placement=self._rank_registry.rank_placement,
            job_status=job_status,
        )
        decision = self._evaluate_detectors(ctx)

        logger.info(
            "loop_tick tick=%d active_run_id=%s decision_action=%s decision_reason=%s",
            self._tick_count, self._rank_registry.active_run_id,
            decision.action.value, decision.reason,
        )

        self._update_exporter_metrics(job_status)
        await self._execute_decision(decision)

    # -------------------------------------------------------------------
    # Internal: detector chain
    # -------------------------------------------------------------------

    def _evaluate_detectors(self, ctx: DetectorContext) -> Decision:
        for detector in self._detectors:
            decision = detector.evaluate(ctx)
            if decision.action != ActionType.NONE:
                return decision

        return _ALL_DETECTORS_PASSED

    # -------------------------------------------------------------------
    # Internal: exporter metric updates
    # -------------------------------------------------------------------

    def _update_exporter_metrics(self, job_status: JobStatus) -> None:
        if self._controller_exporter is None:
            return

        self._controller_exporter.update_training_job_status(job_status)
        self._controller_exporter.update_tick_count()

        is_recovery = self._recovery_orchestrator is not None
        self._controller_exporter.update_mode(is_recovery=is_recovery)
        if not is_recovery:
            self._controller_exporter.update_recovery_phase(None)

        loss = self._mini_wandb.latest(metric_name="loss")
        mfu = self._mini_wandb.latest(metric_name="mfu")
        self._controller_exporter.update_training_metrics(loss=loss, mfu=mfu)

    # -------------------------------------------------------------------
    # Internal: decision execution
    # -------------------------------------------------------------------

    async def _execute_decision(self, decision: Decision) -> None:
        if decision.action == ActionType.NONE:
            return

        if decision.action == ActionType.MARK_BAD_AND_RESTART:
            await handle_mark_bad_and_restart(
                decision=decision,
                node_manager=self._node_manager,
                training_job=self._training_job,
                mini_wandb=self._mini_wandb,
                notifier=self._notifier,
            )
        elif decision.action == ActionType.ENTER_RECOVERY:
            self._recovery_orchestrator = await handle_enter_recovery(
                decision=decision,
                node_manager=self._node_manager,
                training_job=self._training_job,
                metric_store=self._metric_store,
                mini_wandb=self._mini_wandb,
                notifier=self._notifier,
                diagnostic_scheduler=self._diagnostic_scheduler,
                controller_exporter=self._controller_exporter,
            )
        elif decision.action == ActionType.NOTIFY_HUMAN:
            await handle_notify_human(
                decision=decision,
                notifier=self._notifier,
            )
        else:
            raise ValueError(f"Unknown action type: {decision.action}")
