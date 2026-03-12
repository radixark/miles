from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from miles.utils.ft.adapters.types import JobStatus, MainJobProtocol, NodeManagerProtocol, NotifierProtocol
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.node_agents import NodeAgentCoverageChecker, NodeAgentRegistry
from miles.utils.ft.controller.subsystem_hub import SubsystemSpec, TrainingRankRoster
from miles.utils.ft.controller.state_machines.main.models import MainContext, MainState, NormalSt
from miles.utils.ft.utils.box import Box
from miles.utils.ft.controller.state_machines.subsystem import RecoveringSt
from miles.utils.ft.controller.state_machines.recovery import RECOVERY_STATE_TO_INT
from miles.utils.ft.controller.state_machines.utils import safe_notify
from miles.utils.ft.controller.types import (
    DiagnosticOrchestratorProtocol,
    MetricStore,
    SharedDeps,
)
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter
from miles.utils.ft.utils.state_machine import StateMachine

logger = logging.getLogger(__name__)


@dataclass
class TickDeps:
    """Shared dependencies passed by FtController to TickLoop on each tick."""

    main_job: MainJobProtocol
    metric_store: MetricStore
    notifier: NotifierProtocol | None
    node_manager: NodeManagerProtocol
    diagnostic_orchestrator: DiagnosticOrchestratorProtocol
    recovery_timeout_seconds: int
    max_simultaneous_bad_nodes: int
    subsystem_specs: dict[str, SubsystemSpec]
    on_main_job_new_run: Callable[[str], None]
    rank_pids_provider: Callable[[str], dict[int, int]]
    on_recovery_duration: Callable[[float], None] | None
    controller_exporter: ControllerExporter
    registration_grace_ticks: int
    training_rank_roster_box: Box[TrainingRankRoster | None]
    node_agent_registry: NodeAgentRegistry


class TickLoop:
    """Pure tick execution engine — holds only tick-specific state.

    All shared dependencies are passed via TickDeps on each tick() call,
    keeping TickLoop construction lightweight and free of FtController.
    """

    def __init__(
        self,
        *,
        state_machine: StateMachine[MainState, MainContext],
        registration_grace_ticks: int = 5,
    ) -> None:
        self.state_machine = state_machine
        self.tick_count: int = 0
        self._run_start_tick: int = 0
        self._registration_grace_ticks = registration_grace_ticks

        self._detector_crash_tracker = SlidingWindowCounter(window_seconds=1800, threshold=5)
        self._tick_failure_tracker = SlidingWindowCounter(window_seconds=300, threshold=5)
        self._node_agent_coverage_checker = NodeAgentCoverageChecker()

    # ------------------------------------------------------------------
    # Tick execution
    # ------------------------------------------------------------------

    async def tick(self, deps: TickDeps) -> None:
        self.tick_count += 1
        t0 = time.monotonic()
        job_status: JobStatus | None = None
        try:
            roster = deps.training_rank_roster_box.value
            if roster is not None:
                roster.warn_if_incomplete()
                self._node_agent_coverage_checker.check(
                    subsystem_node_ids=set(roster.rank_placement.values()),
                    registered_agent_node_ids=deps.node_agent_registry.registered_node_ids(),
                )
            job_status = await deps.main_job.get_status()

            context = self._build_controller_context(job_status=job_status, deps=deps)
            await self.state_machine.step(context)
        except Exception:
            logger.error("tick_failed tick=%d", self.tick_count, exc_info=True)
            self._tick_failure_tracker.record()
            if self._tick_failure_tracker.should_notify:
                logger.error("tick_persistently_failing: %s", self._tick_failure_tracker.summary())
                await safe_notify(
                    deps.notifier,
                    title="Controller tick persistently failing",
                    content=self._tick_failure_tracker.summary(),
                )
        finally:
            tick_duration = time.monotonic() - t0
            self._update_exporter_metrics(job_status, tick_duration=tick_duration, deps=deps)

    def _handle_main_job_new_run(self, run_id: str, deps: TickDeps) -> None:
        self._run_start_tick = self.tick_count
        deps.on_main_job_new_run(run_id)

    # ------------------------------------------------------------------
    # Context factory
    # ------------------------------------------------------------------

    def _build_controller_context(self, *, job_status: JobStatus, deps: TickDeps) -> MainContext:
        shared = SharedDeps(
            main_job=deps.main_job,
            subsystem_specs=deps.subsystem_specs,
            metric_store=deps.metric_store,
            notifier=deps.notifier,
            node_manager=deps.node_manager,
            diagnostic_orchestrator=deps.diagnostic_orchestrator,
            detector_crash_tracker=self._detector_crash_tracker,
            recovery_timeout_seconds=deps.recovery_timeout_seconds,
            max_simultaneous_bad_nodes=deps.max_simultaneous_bad_nodes,
            on_main_job_new_run=lambda run_id: self._handle_main_job_new_run(run_id, deps),
            rank_pids_provider=deps.rank_pids_provider,
            controller_exporter=deps.controller_exporter,
            on_recovery_duration=deps.on_recovery_duration,
            registration_grace_ticks=self._registration_grace_ticks,
        )
        return MainContext(
            shared=shared,
            tick_count=self.tick_count,
            run_start_tick=self._run_start_tick,
            job_status=job_status,
            node_metadata=deps.node_agent_registry.all_metadata,
        )

    # ------------------------------------------------------------------
    # Exporter metrics
    # ------------------------------------------------------------------

    def _update_exporter_metrics(
        self,
        job_status: JobStatus | None,
        *,
        tick_duration: float,
        deps: TickDeps,
    ) -> None:
        deps.controller_exporter.update_tick_duration(tick_duration)
        deps.controller_exporter.update_last_tick_timestamp(time.time())

        if job_status is None:
            return

        subsystem_modes = self._collect_subsystem_modes(deps)

        deps.controller_exporter.update_from_state(
            job_status=job_status,
            subsystem_modes=subsystem_modes,
            latest_loss=deps.metric_store.mini_wandb.latest(metric_name="loss"),
            latest_mfu=deps.metric_store.mini_wandb.latest(metric_name="mfu"),
        )

    def _collect_subsystem_modes(self, deps: TickDeps) -> dict[str, tuple[bool, int]]:
        from miles.utils.ft.controller.state_machines.main.models import RestartingMainJobSt

        controller_state = self.state_machine.state
        if isinstance(controller_state, NormalSt):
            result: dict[str, tuple[bool, int]] = {}
            for name, sub_state in controller_state.subsystems.items():
                is_recovery = isinstance(sub_state, RecoveringSt)
                phase_int = 0
                if is_recovery:
                    phase_int = RECOVERY_STATE_TO_INT[type(sub_state.recovery)]
                result[name] = (is_recovery, phase_int)
            return result

        if isinstance(controller_state, RestartingMainJobSt):
            modes = {name: (False, 0) for name in deps.subsystem_specs}
            frozen = controller_state.requestor_frozen_state
            if isinstance(frozen, RecoveringSt):
                modes[controller_state.requestor_name] = (
                    True,
                    RECOVERY_STATE_TO_INT[type(frozen.recovery)],
                )
            return modes

        return {name: (False, 0) for name in deps.subsystem_specs}
