from __future__ import annotations

import asyncio
import logging
from typing import Any

from miles.utils.ft.controller.controller_exporter import ControllerExporter
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.diagnostics.scheduler import DiagnosticScheduler
from miles.utils.ft.controller.mini_prometheus.protocol import (
    MetricStoreProtocol,
    ScrapeTargetManagerProtocol,
)
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery_orchestrator import RecoveryOrchestrator
from miles.utils.ft.models import ActionType, Decision, RECOVERY_PHASE_TO_INT
from miles.utils.ft.platform.protocols import (
    DiagnosticSchedulerProtocol,
    JobStatus,
    NodeManagerProtocol,
    NotificationProtocol,
    TrainingJobProtocol,
)

logger = logging.getLogger(__name__)

_JOB_STATUS_TO_NUMERIC: dict[JobStatus, int] = {
    JobStatus.RUNNING: 1,
    JobStatus.STOPPED: 0,
    JobStatus.FAILED: -1,
    JobStatus.PENDING: 2,
}

_ALL_DETECTORS_PASSED = Decision(action=ActionType.NONE, reason="all detectors passed")


class FtController:
    def __init__(
        self,
        node_manager: NodeManagerProtocol,
        training_job: TrainingJobProtocol,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        notifier: NotificationProtocol | None = None,
        detectors: list[BaseFaultDetector] | None = None,
        tick_interval: float = 30.0,
        controller_exporter: ControllerExporter | None = None,
        scrape_target_manager: ScrapeTargetManagerProtocol | None = None,
        diagnostic_scheduler: DiagnosticSchedulerProtocol | None = None,
    ) -> None:
        self._node_manager = node_manager
        self._training_job = training_job
        self._metric_store = metric_store
        self._mini_wandb = mini_wandb
        self._notifier = notifier
        self._detectors: list[BaseFaultDetector] = detectors or []
        self._tick_interval = tick_interval
        self._controller_exporter = controller_exporter
        self._scrape_target_manager = scrape_target_manager
        self._agents: dict[str, Any] = {}

        self._diagnostic_scheduler: DiagnosticSchedulerProtocol = (
            diagnostic_scheduler
            or DiagnosticScheduler(
                agents=self._agents,
                pipeline=["gpu"],
                rank_pids_provider=self._get_rank_pids_for_node,
            )
        )

        self._active_run_id: str | None = None
        self._expected_world_size: int | None = None
        self._rank_placement: dict[int, str] = {}
        self._rank_pids: dict[int, int] = {}
        self._shutting_down: bool = False
        self._tick_count: int = 0

        self._recovery_orchestrator: RecoveryOrchestrator | None = None
        self._diagnosing_nodes: set[str] = set()

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("controller_start tick_interval=%s", self._tick_interval)
        scrape_task = await self._start_scrape_loop()
        try:
            while not self._shutting_down:
                await self._tick()
                if not self._shutting_down:
                    await asyncio.sleep(self._tick_interval)
        finally:
            await self._stop_scrape_loop(scrape_task)
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
            "active_run_id": self._active_run_id,
            "bad_nodes": bad_nodes,
        }

    # -------------------------------------------------------------------
    # Agent management
    # -------------------------------------------------------------------

    def register_agent(self, node_id: str, agent: Any) -> None:
        self._agents[node_id] = agent
        logger.info("agent_registered node_id=%s", node_id)

    def unregister_agent(self, node_id: str) -> None:
        removed = self._agents.pop(node_id, None)
        if removed is not None:
            logger.info("agent_unregistered node_id=%s", node_id)

    # -------------------------------------------------------------------
    # API Called from Agents
    # -------------------------------------------------------------------

    async def log_step(
        self,
        run_id: str,
        rank: int,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        if self._active_run_id is not None and run_id != self._active_run_id:
            logger.debug(
                "log_step_discarded run_id=%s active_run_id=%s",
                run_id, self._active_run_id,
            )
            return

        self._mini_wandb.log_step(
            run_id=run_id,
            rank=rank,
            step=step,
            metrics=metrics,
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
        if run_id != self._active_run_id:
            logger.info(
                "new_run_registered run_id=%s previous_run_id=%s",
                run_id, self._active_run_id,
            )
            self._active_run_id = run_id
            self._expected_world_size = None
            self._mini_wandb.set_active_run_id(run_id)
            self._mini_wandb.clear()
            self._remove_old_scrape_targets()
            self._rank_placement = {}
            self._rank_pids = {}

        self._expected_world_size = world_size
        self._rank_placement[rank] = node_id
        if pid is not None:
            self._rank_pids[rank] = pid
        logger.info(
            "rank_registered run_id=%s rank=%d world_size=%d node_id=%s",
            run_id, rank, world_size, node_id,
        )

        if self._scrape_target_manager is not None:
            self._scrape_target_manager.add_scrape_target(
                target_id=f"rank-{rank}",
                address=exporter_address,
            )

    def _get_rank_pids_for_node(self, node_id: str) -> dict[int, int]:
        return {
            rank: self._rank_pids[rank]
            for rank, nid in self._rank_placement.items()
            if nid == node_id and rank in self._rank_pids
        }

    def _remove_old_scrape_targets(self) -> None:
        if self._scrape_target_manager is not None:
            for old_rank in self._rank_placement:
                self._scrape_target_manager.remove_scrape_target(f"rank-{old_rank}")

    # -------------------------------------------------------------------
    # Main loop tick
    # -------------------------------------------------------------------

    async def _tick(self) -> None:
        self._tick_count += 1

        if (
            self._expected_world_size is not None
            and len(self._rank_placement) < self._expected_world_size
        ):
            logger.warning(
                "incomplete_rank_registration registered=%d expected=%d run_id=%s",
                len(self._rank_placement), self._expected_world_size, self._active_run_id,
            )

        job_status = await self._training_job.get_training_status()

        if self._recovery_orchestrator is not None:
            await self._recovery_orchestrator.step()
            if self._recovery_orchestrator.is_done():
                logger.info("recovery_complete trigger=%s", self._recovery_orchestrator.trigger)
                self._recovery_orchestrator = None
                self._diagnosing_nodes.clear()
            self._update_exporter_metrics(job_status)
            return

        ctx = DetectorContext(
            metric_store=self._metric_store,
            mini_wandb=self._mini_wandb,
            rank_placement=self._rank_placement,
            job_status=job_status,
        )
        decision = self._evaluate_detectors(ctx)

        logger.info(
            "loop_tick tick=%d active_run_id=%s decision_action=%s decision_reason=%s",
            self._tick_count, self._active_run_id,
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

        status_value = _JOB_STATUS_TO_NUMERIC.get(job_status, 0)
        self._controller_exporter.update_training_job_status(status_value)
        self._controller_exporter.update_tick_count()

        if self._recovery_orchestrator is not None:
            self._controller_exporter.update_mode(1)
            phase_int = RECOVERY_PHASE_TO_INT.get(self._recovery_orchestrator.phase, 0)
            self._controller_exporter.update_recovery_phase(phase_int)
        else:
            self._controller_exporter.update_mode(0)
            self._controller_exporter.update_recovery_phase(0)

        loss = self._mini_wandb.latest(metric_name="loss", rank=0)
        mfu = self._mini_wandb.latest(metric_name="mfu", rank=0)
        self._controller_exporter.update_training_metrics(loss=loss, mfu=mfu)

    # -------------------------------------------------------------------
    # Internal: scrape loop lifecycle
    # -------------------------------------------------------------------

    async def _start_scrape_loop(self) -> asyncio.Task[None] | None:
        start_fn = getattr(self._metric_store, "start", None)
        if start_fn is None or not callable(start_fn):
            return None

        async def _run_scrape() -> None:
            try:
                await start_fn()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.error("scrape_loop_crashed", exc_info=True)

        task = asyncio.create_task(_run_scrape())
        logger.info("scrape_loop_started")
        return task

    async def _stop_scrape_loop(self, task: asyncio.Task[None] | None) -> None:
        if task is None:
            return

        stop_fn = getattr(self._metric_store, "stop", None)
        if stop_fn is not None and callable(stop_fn):
            await stop_fn()

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        logger.info("scrape_loop_stopped")

    # -------------------------------------------------------------------
    # Internal: decision execution
    # -------------------------------------------------------------------

    async def _execute_decision(self, decision: Decision) -> None:
        if decision.action == ActionType.NONE:
            return

        if decision.action == ActionType.MARK_BAD_AND_RESTART:
            logger.warning(
                "decision_mark_bad_and_restart bad_node_ids=%s reason=%s",
                decision.bad_node_ids, decision.reason,
            )
            for node_id in decision.bad_node_ids:
                await self._node_manager.mark_node_bad(
                    node_id, reason=decision.reason,
                )
            await self._training_job.stop_training()
            self._mini_wandb.clear()
            await self._training_job.submit_training()
            return

        if decision.action == ActionType.ENTER_RECOVERY:
            logger.warning(
                "decision_enter_recovery trigger=%s reason=%s",
                decision.trigger, decision.reason,
            )
            self._recovery_orchestrator = RecoveryOrchestrator(
                trigger=decision.trigger,
                node_manager=self._node_manager,
                training_job=self._training_job,
                metric_store=self._metric_store,
                mini_wandb=self._mini_wandb,
                notifier=self._notifier,
                diagnostic_scheduler=self._diagnostic_scheduler,
                controller_exporter=self._controller_exporter,
            )
            return

        if decision.action == ActionType.NOTIFY_HUMAN:
            logger.warning(
                "decision_notify_human reason=%s",
                decision.reason,
            )
            if self._notifier is not None:
                try:
                    await self._notifier.send(
                        title="Fault Alert",
                        content=decision.reason,
                        severity="critical",
                    )
                except Exception:
                    logger.exception("notifier_send_failed")
            return

        raise ValueError(f"Unknown action type: {decision.action}")
