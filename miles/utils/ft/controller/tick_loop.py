from __future__ import annotations

import logging
import time
from collections.abc import Callable

from miles.utils.ft.adapters.types import JobStatus, MainJobProtocol, NodeAgentProtocol, NodeManagerProtocol, NotifierProtocol
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.node_agent_coverage import NodeAgentCoverageChecker
from miles.utils.ft.controller.training_rank_roster import TrainingRankRoster
from miles.utils.ft.controller.state_machines.main.models import MainContext, MainState, NormalState
from miles.utils.ft.utils.box import Box
from miles.utils.ft.controller.state_machines.subsystem import Recovering
from miles.utils.ft.controller.state_machines.recovery import RECOVERY_STATE_TO_INT
from miles.utils.ft.controller.state_machines.utils import safe_notify
from miles.utils.ft.controller.subsystem import SubsystemConfig
from miles.utils.ft.controller.types import (
    ControllerMode,
    DiagnosticOrchestratorProtocol,
    MetricStoreProtocol,
)
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter, SlidingWindowThrottle
from miles.utils.ft.utils.state_machine import StateMachine

logger = logging.getLogger(__name__)


class TickLoop:
    def __init__(
        self,
        *,
        state_machine: StateMachine[MainState, MainContext],
        training_rank_roster_box: Box[TrainingRankRoster],
        agents: dict[str, NodeAgentProtocol],
        main_job: MainJobProtocol,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        notifier: NotifierProtocol | None,
        node_manager: NodeManagerProtocol,
        cooldown: SlidingWindowThrottle,
        max_simultaneous_bad_nodes: int,
        diagnostic_orchestrator: DiagnosticOrchestratorProtocol,
        recovery_timeout_seconds: int,
        subsystem_configs: dict[str, SubsystemConfig],
        on_new_run: Callable[[str], None] | None = None,
        rank_pids_provider: Callable[[str], dict[int, int]] | None = None,
        on_recovery_duration: Callable[[float], None] | None = None,
        controller_exporter: ControllerExporter | None = None,
        registration_grace_ticks: int = 5,
    ) -> None:
        self.state_machine = state_machine
        self._training_rank_roster_box = training_rank_roster_box
        self.tick_count: int = 0

        self._agents = agents
        self._main_job = main_job
        self._metric_store = metric_store
        self._mini_wandb = mini_wandb
        self._notifier = notifier
        self._node_manager = node_manager
        self._cooldown = cooldown
        self._max_simultaneous_bad_nodes = max_simultaneous_bad_nodes
        self._diagnostic_orchestrator = diagnostic_orchestrator
        self._recovery_timeout_seconds = recovery_timeout_seconds
        self.subsystem_configs = subsystem_configs
        self._on_new_run = on_new_run
        self._rank_pids_provider = rank_pids_provider
        self._on_recovery_duration = on_recovery_duration
        self._controller_exporter: ControllerExporter = controller_exporter or NullControllerExporter()
        self._registration_grace_ticks = registration_grace_ticks

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
            self._training_rank_roster_box.value.warn_if_incomplete()
            self._node_agent_coverage_checker.check(
                subsystem_node_ids=set(self._training_rank_roster_box.value.rank_placement.values()),
                registered_agent_node_ids=set(self._agents.keys()),
            )
            job_status = await self._main_job.get_job_status()

            context = self._build_controller_context(job_status=job_status)
            await self.state_machine.step(context)
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

    # ------------------------------------------------------------------
    # Context factory
    # ------------------------------------------------------------------

    def _build_controller_context(self, *, job_status: JobStatus) -> MainContext:
        return MainContext(
            main_job=self._main_job,
            subsystem_configs=self.subsystem_configs,
            tick_count=self.tick_count,
            job_status=job_status,
            metric_store=self._metric_store,
            mini_wandb=self._mini_wandb,
            agents=self._agents,
            notifier=self._notifier,
            node_manager=self._node_manager,
            diagnostic_orchestrator=self._diagnostic_orchestrator,
            cooldown=self._cooldown,
            detector_crash_tracker=self._detector_crash_tracker,
            recovery_timeout_seconds=self._recovery_timeout_seconds,
            max_simultaneous_bad_nodes=self._max_simultaneous_bad_nodes,
            on_new_run=self._on_new_run,
            rank_pids_provider=self._rank_pids_provider,
            controller_exporter=self._controller_exporter,
            on_recovery_duration=self._on_recovery_duration,
            registration_grace_ticks=self._registration_grace_ticks,
        )

    # ------------------------------------------------------------------
    # Exporter metrics
    # ------------------------------------------------------------------

    def _update_exporter_metrics(self, job_status: JobStatus | None, *, tick_duration: float) -> None:
        self._controller_exporter.update_tick_duration(tick_duration)
        self._controller_exporter.update_last_tick_timestamp(time.time())

        if job_status is None:
            return

        main_state = self._extract_main_state()
        is_recovery = isinstance(main_state, Recovering)
        phase_int = 0
        if is_recovery and isinstance(main_state, Recovering):
            phase_int = RECOVERY_STATE_TO_INT.get(type(main_state.recovery), 0)

        self._controller_exporter.update_from_state(
            job_status=job_status,
            mode=ControllerMode.RECOVERY if is_recovery else ControllerMode.MONITORING,
            recovery_phase_int=phase_int,
            latest_loss=self._mini_wandb.latest(metric_name="loss"),
            latest_mfu=self._mini_wandb.latest(metric_name="mfu"),
        )

    def _extract_main_state(self) -> object:
        controller_state = self.state_machine.state
        if isinstance(controller_state, NormalState):
            return controller_state.subsystems.get("training")
        return None
