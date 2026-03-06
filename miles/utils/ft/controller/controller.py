from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterator

from miles.utils.ft.controller.actions import (
    PlatformDeps,
    handle_mark_bad_and_restart,
    handle_notify_human,
)
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator
from miles.utils.ft.controller.metrics.lifecycle import start_metric_store_task, stop_metric_store_task
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.rank_roster import RankRoster
from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle
from miles.utils.ft.controller.lifecycle_manager import RecoveryLifecycleManager
from miles.utils.ft.models.fault import ActionType, Decision
from miles.utils.ft.models.recovery import (
    ControllerMode,
    ControllerStatus,
)
from miles.utils.ft.protocols.agents import NodeAgentProtocol
from miles.utils.ft.protocols.metrics import MetricStoreProtocol, ScrapeTargetManagerProtocol
from miles.utils.ft.protocols.platform import (
    DiagnosticOrchestratorProtocol,
    JobStatus,
    NodeManagerProtocol,
    NotificationProtocol,
    TrainingJobProtocol,
)

logger = logging.getLogger(__name__)

_ALL_DETECTORS_PASSED = Decision.no_fault(reason="all detectors passed")


class FtController:
    def __init__(
        self,
        *,
        platform_deps: PlatformDeps,
        recovery_manager: RecoveryLifecycleManager,
        rank_roster: RankRoster,
        mini_wandb: MiniWandb,
        scrape_target_manager: ScrapeTargetManagerProtocol | None,
        agents: dict[str, NodeAgentProtocol],
        metric_store: MetricStoreProtocol,
        detectors: list[BaseFaultDetector],
        tick_interval: float,
        controller_exporter: ControllerExporter | None = None,
        registration_grace_ticks: int = 5,
    ) -> None:
        self._training_job = platform_deps.training_job
        self._metric_store = metric_store
        self._rank_roster = rank_roster
        self._mini_wandb = mini_wandb
        self._scrape_target_manager = scrape_target_manager
        self._agents = agents
        self._detectors = detectors
        self._tick_interval = tick_interval
        self._controller_exporter = controller_exporter or NullControllerExporter()
        self._registration_grace_ticks = registration_grace_ticks
        self._platform_deps = platform_deps
        self._recovery_manager = recovery_manager

        self._shutting_down: bool = False
        self._tick_count: int = 0

    @classmethod
    def create(
        cls,
        node_manager: NodeManagerProtocol,
        training_job: TrainingJobProtocol,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        scrape_target_manager: ScrapeTargetManagerProtocol | None = None,
        notifier: NotificationProtocol | None = None,
        detectors: list[BaseFaultDetector] | None = None,
        tick_interval: float = 30.0,
        controller_exporter: ControllerExporter | None = None,
        diagnostic_orchestrator: DiagnosticOrchestratorProtocol | None = None,
        recovery_cooldown: SlidingWindowThrottle | None = None,
        registration_grace_ticks: int = 5,
    ) -> FtController:
        agents: dict[str, NodeAgentProtocol] = {}
        rank_roster = RankRoster(scrape_target_manager=scrape_target_manager)

        resolved_orchestrator: DiagnosticOrchestratorProtocol = (
            diagnostic_orchestrator
            or DiagnosticOrchestrator(
                agents=agents,
                pipeline=["gpu"],
            )
        )

        platform_deps = PlatformDeps(
            node_manager=node_manager,
            training_job=training_job,
            metric_store=metric_store,
            mini_wandb=mini_wandb,
            notifier=notifier,
            diagnostic_orchestrator=resolved_orchestrator,
            controller_exporter=controller_exporter,
            on_new_run=None,
        )

        resolved_exporter = controller_exporter or NullControllerExporter()
        duration_cb = resolved_exporter.observe_recovery_duration
        recovery_manager = RecoveryLifecycleManager(
            cooldown=recovery_cooldown or SlidingWindowThrottle(window_minutes=30.0, max_count=3),
            on_recovery_duration=duration_cb,
        )

        instance = cls(
            platform_deps=platform_deps,
            recovery_manager=recovery_manager,
            rank_roster=rank_roster,
            mini_wandb=mini_wandb,
            scrape_target_manager=scrape_target_manager,
            agents=agents,
            metric_store=metric_store,
            detectors=detectors or [],
            tick_interval=tick_interval,
            controller_exporter=controller_exporter,
            registration_grace_ticks=registration_grace_ticks,
        )

        platform_deps.on_new_run = instance._activate_run
        platform_deps.rank_pids_provider = lambda node_id: instance._rank_roster.get_rank_pids_for_node(node_id)
        return instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def rank_roster(self) -> RankRoster:
        return self._rank_roster

    @property
    def mini_wandb(self) -> MiniWandb:
        return self._mini_wandb

    def register_node_agent(
        self, node_id: str, agent: NodeAgentProtocol, exporter_address: str = "",
    ) -> None:
        self._agents[node_id] = agent
        if exporter_address and self._scrape_target_manager is not None:
            self._scrape_target_manager.add_scrape_target(
                target_id=node_id, address=exporter_address,
            )
        logger.info("agent_registered node_id=%s exporter=%s", node_id, exporter_address)

    async def submit_initial_training(self) -> str:
        run_id = await self._training_job.submit_training()
        self._activate_run(run_id)
        return run_id

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
        snap = self._recovery_manager.snapshot()

        iteration_val = self._mini_wandb.latest(metric_name="iteration")
        latest_iteration = int(iteration_val) if iteration_val is not None else None

        return ControllerStatus(
            mode=ControllerMode.RECOVERY if snap.in_progress else ControllerMode.MONITORING,
            recovery_phase=snap.phase,
            phase_history=snap.phase_history,
            tick_count=self._tick_count,
            active_run_id=self._rank_roster.run_id,
            bad_nodes=snap.diagnosing_nodes,
            recovery_in_progress=snap.in_progress,
            bad_nodes_confirmed=snap.bad_nodes_confirmed,
            latest_iteration=latest_iteration,
        )

    def _activate_run(self, run_id: str) -> None:
        """Create a fresh RankRoster for the new run and switch MiniWandb."""
        self._rank_roster.cleanup()
        self._rank_roster = RankRoster(
            run_id=run_id,
            scrape_target_manager=self._scrape_target_manager,
        )
        self._mini_wandb.set_active_run_id(run_id)
        logger.info("run_activated run_id=%s", run_id)

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        self._tick_count += 1
        t0 = time.monotonic()
        job_status: JobStatus | None = None
        try:
            self._rank_roster.warn_if_incomplete()
            job_status = await self._training_job.get_training_status()
            await self._tick_inner(job_status)
        except Exception:
            logger.error("tick_failed tick=%d", self._tick_count, exc_info=True)
        finally:
            tick_duration = time.monotonic() - t0
            self._update_exporter_metrics(job_status, tick_duration=tick_duration)

    async def _tick_inner(self, job_status: JobStatus) -> None:
        if self._recovery_manager.in_progress:
            new_bad_nodes = self._collect_critical_bad_nodes(job_status)
            if new_bad_nodes:
                self._recovery_manager.add_bad_nodes(new_bad_nodes)
            await self._recovery_manager.step()
            return

        if not self._should_run_detectors():
            return

        ctx = self._build_detector_context(job_status)
        decision = self._run_detectors(ctx)

        logger.info(
            "loop_tick tick=%d active_run_id=%s decision_action=%s decision_reason=%s",
            self._tick_count, self._rank_roster.run_id,
            decision.action.value, decision.reason,
        )

        await self._execute_decision(decision)

    def _should_run_detectors(self) -> bool:
        if len(self._rank_roster.rank_placement) == 0:
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
            rank_placement=dict(self._rank_roster.rank_placement),
            job_status=job_status,
        )

    def _run_detectors(self, ctx: DetectorContext) -> Decision:
        for decision in self._run_detectors_raw(ctx):
            if decision.action != ActionType.NONE:
                return decision

        return _ALL_DETECTORS_PASSED

    def _collect_critical_bad_nodes(self, job_status: JobStatus) -> set[str]:
        """Run is_critical detectors and return any newly discovered bad node ids."""
        ctx = self._build_detector_context(job_status)
        bad_nodes: set[str] = set()

        for decision in self._run_detectors_raw(ctx, critical_only=True):
            if decision.action == ActionType.MARK_BAD_AND_RESTART and decision.bad_node_ids:
                bad_nodes.update(decision.bad_node_ids)

        return bad_nodes

    def _run_detectors_raw(
        self,
        ctx: DetectorContext,
        *,
        critical_only: bool = False,
    ) -> Iterator[Decision]:
        for detector in self._detectors:
            if critical_only and not detector.is_critical:
                continue
            try:
                yield detector.evaluate(ctx)
            except Exception:
                logger.error(
                    "detector_evaluate_failed detector=%s",
                    type(detector).__name__,
                    exc_info=True,
                )

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
            self._rank_roster.run_id, self._tick_count,
        )
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

    def _update_exporter_metrics(self, job_status: JobStatus | None, *, tick_duration: float) -> None:
        self._controller_exporter.update_tick_duration(tick_duration)
        self._controller_exporter.update_last_tick_timestamp(time.time())

        if job_status is None:
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
        self._controller_exporter.stop()
        if self._platform_deps.notifier is not None:
            try:
                await self._platform_deps.notifier.aclose()
            except Exception:
                logger.warning("notifier_aclose_failed", exc_info=True)
