from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime

from miles.utils.ft.adapters.types import JobStatus, MainJobProtocol, NodeAgentProtocol, NotifierProtocol
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.node_agent_coverage import NodeAgentCoverageChecker
from miles.utils.ft.controller.rank_roster import RankRoster
from miles.utils.ft.controller.state_machines.main import MainContext, MainState, Recovering
from miles.utils.ft.controller.state_machines.recovery import RECOVERY_STATE_TO_INT, RecoveryContext
from miles.utils.ft.controller.state_machines.restart import RestartContext
from miles.utils.ft.controller.state_machines.utils import safe_notify
from miles.utils.ft.controller.types import (
    ControllerMode,
    DiagnosticOrchestratorProtocol,
    MetricStoreProtocol,
    TriggerType,
)
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter, SlidingWindowThrottle
from miles.utils.ft.utils.state_machine import StateMachine, StateMachineStepper

logger = logging.getLogger(__name__)


class TickLoop:
    def __init__(
        self,
        *,
        state_machine: StateMachine[MainState, MainContext],
        rank_roster: RankRoster,
        agents: dict[str, NodeAgentProtocol],
        main_job: MainJobProtocol,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        detectors: list[BaseFaultDetector],
        notifier: NotifierProtocol | None,
        cooldown: SlidingWindowThrottle,
        recovery_stepper: StateMachineStepper,
        on_recovery_duration: Callable[[float], None] | None,
        max_simultaneous_bad_nodes: int,
        diagnostic_orchestrator: DiagnosticOrchestratorProtocol,
        restart_stepper: StateMachineStepper,
        restart_context: RestartContext | None,
        recovery_timeout_seconds: int,
        controller_exporter: ControllerExporter | None = None,
        registration_grace_ticks: int = 5,
    ) -> None:
        self.state_machine = state_machine
        self.rank_roster = rank_roster
        self.tick_count: int = 0

        self._agents = agents
        self._main_job = main_job
        self._metric_store = metric_store
        self._mini_wandb = mini_wandb
        self._detectors = detectors
        self._controller_exporter: ControllerExporter = controller_exporter or NullControllerExporter()
        self._registration_grace_ticks = registration_grace_ticks

        self._notifier = notifier
        self._cooldown = cooldown
        self._recovery_stepper = recovery_stepper
        self._on_recovery_duration = on_recovery_duration
        self._max_simultaneous_bad_nodes = max_simultaneous_bad_nodes
        self._diagnostic_orchestrator = diagnostic_orchestrator
        self._restart_stepper = restart_stepper
        self._restart_context = restart_context
        self._recovery_timeout_seconds = recovery_timeout_seconds

        self._detector_crash_tracker = SlidingWindowCounter(window_seconds=1800, threshold=5)
        self._tick_failure_tracker = SlidingWindowCounter(window_seconds=300, threshold=5)
        self._node_agent_coverage_checker = NodeAgentCoverageChecker()

    # ------------------------------------------------------------------
    # Tick execution
    # ------------------------------------------------------------------

    async def tick(self) -> None:
        self.tick_count += 1
        t0 = time.monotonic()
        job_status: JobStatus | None = None
        try:
            self.rank_roster.warn_if_incomplete()
            self._node_agent_coverage_checker.check(
                training_node_ids=set(self.rank_roster.rank_placement.values()),
                registered_agent_node_ids=set(self._agents.keys()),
            )
            job_status = await self._main_job.get_job_status()

            should_run = self._should_run_detectors()
            detector_ctx = self._build_detector_context(job_status) if should_run else None

            main_context = self._build_main_context(
                job_status=job_status,
                should_run_detectors=should_run,
                detector_context=detector_ctx,
            )

            await self.state_machine.step(main_context)
        except Exception:
            logger.error("tick_failed tick=%d", self.tick_count, exc_info=True)
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
        if len(self.rank_roster.rank_placement) == 0:
            logger.info("skip_detectors_no_ranks tick=%d", self.tick_count)
            return False

        if self.tick_count <= self._registration_grace_ticks:
            logger.info(
                "skip_detectors_grace_period tick=%d grace_ticks=%d",
                self.tick_count,
                self._registration_grace_ticks,
            )
            return False

        return True

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
            rank_pids_provider=lambda node_id: self.rank_roster.get_rank_pids_for_node(node_id),
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
            tick_count=self.tick_count,
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

    def _build_detector_context(self, job_status: JobStatus) -> DetectorContext:
        return DetectorContext(
            metric_store=self._metric_store,
            mini_wandb=self._mini_wandb,
            rank_placement=dict(self.rank_roster.rank_placement),
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

        is_recovery = isinstance(self.state_machine.state, Recovering)
        phase_int = 0
        if is_recovery:
            state = self.state_machine.state
            if isinstance(state, Recovering):
                phase_int = RECOVERY_STATE_TO_INT.get(type(state.recovery), 0)

        self._controller_exporter.update_from_state(
            job_status=job_status,
            mode=ControllerMode.RECOVERY if is_recovery else ControllerMode.MONITORING,
            recovery_phase_int=phase_int,
            latest_loss=self._mini_wandb.latest(metric_name="loss"),
            latest_mfu=self._mini_wandb.latest(metric_name="mfu"),
        )
