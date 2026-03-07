from __future__ import annotations

import asyncio
import logging
import time

from miles.utils.ft.controller.actions import PlatformDeps, handle_notify_human
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator
from miles.utils.ft.controller.main_state_machine import (
    DetectingAnomaly,
    MainStepper,
    MainState,
    Recovering,
)
from miles.utils.ft.controller.metrics.lifecycle import start_metric_store_task, stop_metric_store_task
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.rank_roster import RankRoster
from miles.utils.ft.controller.recovery.alert_checker import AlertChecker
from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle
from miles.utils.ft.controller.recovery.recovery_stepper import (
    BAD_NODES_CONFIRMED_TYPES,
    DirectlyRestarting,
    EvictingAndRestarting,
    RECOVERY_STATE_TO_INT,
    RecoveryStepper,
    RecoveryState,
)
from miles.utils.ft.controller.recovery.restart_stepper import RestartStepper
from miles.utils.ft.controller.state_machine import StateMachine
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


def _recovery_phase_name(recovery: RecoveryState) -> str:
    if isinstance(recovery, (EvictingAndRestarting, DirectlyRestarting)):
        return type(recovery.restart).__name__
    return type(recovery).__name__


class FtController:
    def __init__(
        self,
        *,
        platform_deps: PlatformDeps,
        state_machine: StateMachine[MainState],
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
        self._state_machine = state_machine

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
        max_simultaneous_bad_nodes: int = 3,
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
        cooldown = recovery_cooldown or SlidingWindowThrottle(window_minutes=30.0, max_count=3)

        restart_stepper = RestartStepper(
            node_manager=node_manager,
            training_job=training_job,
            mini_wandb=mini_wandb,
            notifier=notifier,
            on_new_run=None,
            monitoring_success_iterations=10,
            monitoring_timeout_seconds=600,
        )

        recovery_stepper = RecoveryStepper(
            alert_checker=AlertChecker(metric_store=metric_store),
            diagnostic_orchestrator=resolved_orchestrator,
            restart_stepper=restart_stepper,
            notifier=notifier,
            timeout_seconds=1800,
        )

        main_stepper = MainStepper(
            platform_deps=platform_deps,
            recovery_stepper=recovery_stepper,
            detectors=detectors or [],
            cooldown=cooldown,
            on_recovery_duration=duration_cb,
            max_simultaneous_bad_nodes=max_simultaneous_bad_nodes,
        )

        state_machine: StateMachine[MainState] = StateMachine(
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
            controller_exporter=controller_exporter,
            registration_grace_ticks=registration_grace_ticks,
        )

        platform_deps.on_new_run = instance._activate_run
        platform_deps.rank_pids_provider = lambda node_id: instance._rank_roster.get_rank_pids_for_node(node_id)

        restart_stepper.set_on_new_run(instance._activate_run)
        recovery_stepper.set_rank_pids_provider(
            lambda node_id: instance._rank_roster.get_rank_pids_for_node(node_id),
        )

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
        state = self._state_machine.state
        main_stepper: MainStepper = self._state_machine.stepper  # type: ignore[assignment]
        iteration_val = self._mini_wandb.latest(metric_name="iteration")
        latest_iteration = int(iteration_val) if iteration_val is not None else None

        phase_history = self._build_phase_history()

        if isinstance(state, Recovering):
            recovery = state.recovery
            mode = ControllerMode.RECOVERY
            recovery_phase_str = _recovery_phase_name(recovery)
            bad_nodes = sorted(main_stepper._get_known_bad_nodes(recovery))
            bad_nodes_confirmed = type(recovery) in BAD_NODES_CONFIRMED_TYPES
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

            main_stepper: MainStepper = self._state_machine.stepper  # type: ignore[assignment]
            main_stepper.set_tick_context(
                job_status=job_status,
                tick_count=self._tick_count,
                should_run_detectors=should_run,
                detector_context=detector_ctx,
            )

            await self._state_machine.step()
        except Exception:
            logger.error("tick_failed tick=%d", self._tick_count, exc_info=True)
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
                self._tick_count, self._registration_grace_ticks,
            )
            return False

        return True

    def _build_detector_context(self, job_status: JobStatus):
        from miles.utils.ft.controller.detectors.base import DetectorContext
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

        state = self._state_machine.state
        is_recovery = isinstance(state, Recovering)
        phase_int = 0
        if is_recovery:
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
