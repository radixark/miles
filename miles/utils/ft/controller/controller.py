from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.state_machines.main import (
    DetectingAnomaly,
    MainContext,
    MainState,
    Recovering,
    create_main_stepper,
    get_known_bad_nodes,
)
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.metrics.lifecycle import start_metric_store_task, stop_metric_store_task
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.rank_roster import RankRoster
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter, SlidingWindowThrottle
from miles.utils.ft.controller.state_machines.recovery import (
    RECOVERY_STATE_TO_INT,
    RECOVERY_TIMEOUT_SECONDS,
    EvictingAndRestarting,
    NotifyHumans,
    RecoveryContext,
    RecoveryDone,
    RecoveryState,
    create_recovery_stepper,
)
from miles.utils.ft.controller.state_machines.restart import RestartContext, create_restart_stepper
from miles.utils.ft.controller.state_machines.utils import safe_notify
from miles.utils.ft.models.fault import TriggerType
from miles.utils.ft.models.recovery import ControllerMode, ControllerStatus
from miles.utils.ft.protocols.agents import NodeAgentProtocol
from miles.utils.ft.protocols.metrics import MetricQueryProtocol, MetricStoreProtocol, ScrapeTargetManagerProtocol
from miles.utils.ft.protocols.controller import DiagnosticOrchestratorProtocol
from miles.utils.ft.protocols.platform import (
    JobStatus,
    NodeManagerProtocol,
    NotifierProtocol,
    TrainingJobProtocol,
)
from miles.utils.ft.utils.state_machine import StateMachine, StateMachineStepper

logger = logging.getLogger(__name__)


@dataclass
class PlatformDeps:
    """Bundles platform-level dependencies shared across action handlers."""

    node_manager: NodeManagerProtocol
    training_job: TrainingJobProtocol
    metric_store: MetricQueryProtocol
    mini_wandb: MiniWandb
    notifier: NotifierProtocol | None
    diagnostic_orchestrator: DiagnosticOrchestratorProtocol
    controller_exporter: ControllerExporter | None
    on_new_run: Callable[[str], None] | None = field(default=None)
    rank_pids_provider: Callable[[str], dict[int, int]] | None = field(default=None)


def _recovery_phase_name(recovery: RecoveryState) -> str:
    if isinstance(recovery, EvictingAndRestarting):
        return type(recovery.restart).__name__
    return type(recovery).__name__


class FtController:
    def __init__(
        self,
        *,
        platform_deps: PlatformDeps,
        state_machine: StateMachine[MainState, MainContext],
        rank_roster: RankRoster,
        mini_wandb: MiniWandb,
        scrape_target_manager: ScrapeTargetManagerProtocol | None,
        agents: dict[str, NodeAgentProtocol],
        metric_store: MetricStoreProtocol,
        detectors: list[BaseFaultDetector],
        tick_interval: float,
        # deps for MainContext building
        notifier: NotifierProtocol | None,
        cooldown: SlidingWindowThrottle,
        recovery_stepper: StateMachineStepper,
        on_recovery_duration: Callable[[float], None] | None,
        max_simultaneous_bad_nodes: int,
        # deps for RecoveryContext building
        diagnostic_orchestrator: DiagnosticOrchestratorProtocol,
        restart_stepper: StateMachineStepper,
        recovery_timeout_seconds: int,
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
        self._state_machine = state_machine

        self._notifier = notifier
        self._cooldown = cooldown
        self._recovery_stepper = recovery_stepper
        self._on_recovery_duration = on_recovery_duration
        self._max_simultaneous_bad_nodes = max_simultaneous_bad_nodes
        self._diagnostic_orchestrator = diagnostic_orchestrator
        self._restart_stepper = restart_stepper
        self._recovery_timeout_seconds = recovery_timeout_seconds

        self._restart_context: RestartContext | None = None
        self._detector_crash_tracker = SlidingWindowCounter(window_seconds=1800, threshold=5)
        self._tick_failure_tracker = SlidingWindowCounter(window_seconds=300, threshold=5)

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
        notifier: NotifierProtocol | None = None,
        detectors: list[BaseFaultDetector] | None = None,
        tick_interval: float = 30.0,
        controller_exporter: ControllerExporter | None = None,
        diagnostic_orchestrator: DiagnosticOrchestratorProtocol | None = None,
        recovery_cooldown: SlidingWindowThrottle | None = None,
        registration_grace_ticks: int = 5,
        max_simultaneous_bad_nodes: int = 3,
        monitoring_success_iterations: int = 10,
        monitoring_timeout_seconds: int = 600,
        recovery_timeout_seconds: int = RECOVERY_TIMEOUT_SECONDS,
    ) -> FtController:
        from miles.utils.ft.controller.diagnostics.executors import build_all_cluster_executors
        from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator

        agents: dict[str, NodeAgentProtocol] = {}
        rank_roster = RankRoster(scrape_target_manager=scrape_target_manager)

        resolved_orchestrator: DiagnosticOrchestratorProtocol = diagnostic_orchestrator or DiagnosticOrchestrator(
            agents=agents,
            pipeline=list(build_all_cluster_executors().values()),
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
        cooldown = recovery_cooldown or SlidingWindowThrottle(window_minutes=30.0, max_count=3)

        restart_stepper = create_restart_stepper()
        recovery_stepper = create_recovery_stepper()
        main_stepper = create_main_stepper()

        state_machine: StateMachine[MainState, MainContext] = StateMachine(
            initial_state=DetectingAnomaly(),
            stepper=main_stepper,
        )

        instance = cls(
            platform_deps=platform_deps,
            state_machine=state_machine,
            rank_roster=rank_roster,
            mini_wandb=mini_wandb,
            scrape_target_manager=scrape_target_manager,
            agents=agents,
            metric_store=metric_store,
            detectors=detectors or [],
            tick_interval=tick_interval,
            notifier=notifier,
            cooldown=cooldown,
            recovery_stepper=recovery_stepper,
            on_recovery_duration=duration_cb,
            max_simultaneous_bad_nodes=max_simultaneous_bad_nodes,
            diagnostic_orchestrator=resolved_orchestrator,
            restart_stepper=restart_stepper,
            recovery_timeout_seconds=recovery_timeout_seconds,
            controller_exporter=controller_exporter,
            registration_grace_ticks=registration_grace_ticks,
        )

        instance._restart_context = RestartContext(
            node_manager=node_manager,
            training_job=training_job,
            mini_wandb=mini_wandb,
            notifier=notifier,
            on_new_run=instance._activate_run,
            monitoring_success_iterations=monitoring_success_iterations,
            monitoring_timeout_seconds=monitoring_timeout_seconds,
        )

        platform_deps.on_new_run = instance._activate_run
        platform_deps.rank_pids_provider = lambda node_id: instance._rank_roster.get_rank_pids_for_node(node_id)

        return instance

    # ------------------------------------------------------------------
    # Context factories
    # ------------------------------------------------------------------

    def _build_recovery_context(
        self,
        trigger: TriggerType,
        recovery_start_time: datetime,
    ) -> RecoveryContext:
        return RecoveryContext(
            trigger=trigger,
            recovery_start_time=recovery_start_time,
            diagnostic_orchestrator=self._diagnostic_orchestrator,
            restart_stepper=self._restart_stepper,
            restart_context=self._restart_context,
            notifier=self._notifier,
            timeout_seconds=self._recovery_timeout_seconds,
            rank_pids_provider=lambda node_id: self._rank_roster.get_rank_pids_for_node(node_id),
        )

    def _build_main_context(
        self,
        *,
        job_status: JobStatus,
        should_run_detectors: bool,
        detector_context: DetectorContext | None,
    ) -> MainContext:
        return MainContext(
            job_status=job_status,
            tick_count=self._tick_count,
            should_run_detectors=should_run_detectors,
            detector_context=detector_context,
            notifier=self._notifier,
            detectors=self._detectors,
            cooldown=self._cooldown,
            detector_crash_tracker=self._detector_crash_tracker,
            recovery_stepper=self._recovery_stepper,
            recovery_context_factory=self._build_recovery_context,
            on_recovery_duration=self._on_recovery_duration,
            max_simultaneous_bad_nodes=self._max_simultaneous_bad_nodes,
        )

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
        self,
        node_id: str,
        agent: NodeAgentProtocol,
        exporter_address: str = "",
    ) -> None:
        self._agents[node_id] = agent
        if exporter_address and self._scrape_target_manager is not None:
            self._scrape_target_manager.add_scrape_target(
                target_id=node_id,
                address=exporter_address,
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
        state = self._state_machine.state
        iteration_val = self._mini_wandb.latest(metric_name="iteration")
        latest_iteration = int(iteration_val) if iteration_val is not None else None

        phase_history = self._build_phase_history()

        if isinstance(state, Recovering):
            recovery = state.recovery
            mode = ControllerMode.RECOVERY
            recovery_phase_str = _recovery_phase_name(recovery)
            bad_nodes = sorted(get_known_bad_nodes(recovery))
            bad_nodes_confirmed = bool(bad_nodes) or isinstance(recovery, (NotifyHumans, RecoveryDone))
        else:
            mode = ControllerMode.MONITORING
            recovery_phase_str = None
            bad_nodes = []
            bad_nodes_confirmed = False

        return ControllerStatus(
            mode=mode,
            recovery_phase=recovery_phase_str,
            phase_history=phase_history if phase_history else None,
            tick_count=self._tick_count,
            active_run_id=self._rank_roster.run_id,
            bad_nodes=bad_nodes,
            recovery_in_progress=isinstance(state, Recovering),
            bad_nodes_confirmed=bad_nodes_confirmed,
            latest_iteration=latest_iteration,
        )

    def _build_phase_history(self) -> list[str]:
        phases: list[str] = []
        for past_state in self._state_machine.state_history:
            if isinstance(past_state, Recovering):
                name = _recovery_phase_name(past_state.recovery)
                if not phases or phases[-1] != name:
                    phases.append(name)
        return phases

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

            should_run = self._should_run_detectors()
            detector_ctx = self._build_detector_context(job_status) if should_run else None

            main_context = self._build_main_context(
                job_status=job_status,
                should_run_detectors=should_run,
                detector_context=detector_ctx,
            )

            await self._state_machine.step(main_context)
        except Exception:
            logger.error("tick_failed tick=%d", self._tick_count, exc_info=True)
            self._tick_failure_tracker.record()
            if self._tick_failure_tracker.should_notify:
                logger.error("tick_persistently_failing: %s", self._tick_failure_tracker.summary())
                await safe_notify(
                    self._notifier,
                    title="Controller tick persistently failing",
                    content=self._tick_failure_tracker.summary(),
                )
        finally:
            tick_duration = time.monotonic() - t0
            self._update_exporter_metrics(job_status, tick_duration=tick_duration)

    def _should_run_detectors(self) -> bool:
        if len(self._rank_roster.rank_placement) == 0:
            logger.info("skip_detectors_no_ranks tick=%d", self._tick_count)
            return False

        if self._tick_count <= self._registration_grace_ticks:
            logger.info(
                "skip_detectors_grace_period tick=%d grace_ticks=%d",
                self._tick_count,
                self._registration_grace_ticks,
            )
            return False

        return True

    def _build_detector_context(self, job_status: JobStatus) -> DetectorContext:
        return DetectorContext(
            metric_store=self._metric_store,
            mini_wandb=self._mini_wandb,
            rank_placement=dict(self._rank_roster.rank_placement),
            job_status=job_status,
        )

    # ------------------------------------------------------------------
    # Exporter metrics
    # ------------------------------------------------------------------

    def _update_exporter_metrics(self, job_status: JobStatus | None, *, tick_duration: float) -> None:
        self._controller_exporter.update_tick_duration(tick_duration)
        self._controller_exporter.update_last_tick_timestamp(time.time())

        if job_status is None:
            return

        is_recovery = isinstance(self._state_machine.state, Recovering)
        phase_int = 0
        if is_recovery:
            state = self._state_machine.state
            if isinstance(state, Recovering):
                phase_int = RECOVERY_STATE_TO_INT.get(type(state.recovery), 0)

        self._controller_exporter.update_from_state(
            job_status=job_status,
            mode=ControllerMode.RECOVERY if is_recovery else ControllerMode.MONITORING,
            recovery_phase_int=phase_int,
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
