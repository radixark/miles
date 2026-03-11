from __future__ import annotations

import logging
from datetime import datetime, timezone

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.detectors.base import DetectorContext
from miles.utils.ft.controller.state_machines.main.models import (
    MainContext,
    MainState,
    NormalState,
    RestartingMainJobState,
)
from miles.utils.ft.controller.state_machines.subsystem.models import (
    SubsystemContext,
    SubsystemState,
    Recovering,
)
from miles.utils.ft.controller.state_machines.recovery.models import (
    EvictingAndRestarting,
    RecoveryContext,
)
from miles.utils.ft.controller.state_machines.restart.models import (
    RestartContext,
    RestartingMainJob as RestartingMainJobRestart,
)
from miles.utils.ft.controller.state_machines.recovery import create_recovery_stepper
from miles.utils.ft.controller.state_machines.restart import create_restart_stepper
from miles.utils.ft.controller.state_machines.utils import safe_notify
from miles.utils.ft.controller.subsystem import SubsystemEntry
from miles.utils.ft.controller.types import TriggerType
from miles.utils.ft.utils.state_machine import StateHandler

logger = logging.getLogger(__name__)


def _find_restart_requestor(subsystems: dict[str, SubsystemEntry]) -> str | None:
    for name, entry in subsystems.items():
        sm_state = entry.state_machine.state
        if not isinstance(sm_state, Recovering):
            continue
        if not isinstance(sm_state.recovery, EvictingAndRestarting):
            continue
        if isinstance(sm_state.recovery.restart, RestartingMainJobRestart) and not sm_state.recovery.restart.externally_fulfilled:
            return name
    return None


def _update_externally_fulfilled(frozen_state: SubsystemState) -> SubsystemState:
    assert isinstance(frozen_state, Recovering)
    recovery = frozen_state.recovery
    assert isinstance(recovery, EvictingAndRestarting)
    restart = recovery.restart
    assert isinstance(restart, RestartingMainJobRestart)
    new_restart = restart.model_copy(update={"externally_fulfilled": True})
    new_recovery = recovery.model_copy(update={"restart": new_restart})
    return frozen_state.model_copy(update={"recovery": new_recovery})


class NormalStateHandler(StateHandler[NormalState, MainContext]):
    def __init__(self) -> None:
        self._restart_stepper = create_restart_stepper()
        self._recovery_stepper = create_recovery_stepper()

    async def step(self, state: NormalState, context: MainContext) -> MainState | None:
        # Step 1: Step all sub-SMs
        for name, entry in state.subsystems.items():
            sub_ctx = self._build_subsystem_context(entry=entry, context=context)
            await entry.state_machine.step(sub_ctx)

        # Step 2: Peek into sub-SMs for RestartingMainJob(externally_fulfilled=False)
        requestor = _find_restart_requestor(state.subsystems)
        if requestor is not None:
            frozen_state = state.subsystems[requestor].state_machine.state
            logger.info("sub-SM %r requested main job restart (peek-and-freeze)", requestor)
            await context.main_job.stop_job()
            run_id = await context.main_job.submit_job()
            if context.on_new_run is not None:
                context.on_new_run(run_id)
            return RestartingMainJobState(
                requestor_name=requestor,
                start_time=datetime.now(timezone.utc),
                requestor_frozen_state=frozen_state,
            )

        return None

    def _build_subsystem_context(
        self,
        *,
        entry: SubsystemEntry,
        context: MainContext,
    ) -> SubsystemContext:
        should_run = self._should_run_detectors(entry=entry, context=context)
        detector_ctx = self._build_detector_context(entry=entry, context=context) if should_run else None

        return SubsystemContext(
            job_status=context.job_status,
            tick_count=context.tick_count,
            should_run_detectors=should_run,
            detector_context=detector_ctx,
            notifier=context.notifier,
            detectors=entry.detectors,
            cooldown=context.cooldown,
            detector_crash_tracker=context.detector_crash_tracker,
            recovery_stepper=self._recovery_stepper,
            recovery_context_factory=lambda trigger, start_time: self._build_recovery_context(
                entry=entry,
                context=context,
                trigger=trigger,
                recovery_start_time=start_time,
            ),
            on_recovery_duration=context.on_recovery_duration,
            max_simultaneous_bad_nodes=context.max_simultaneous_bad_nodes,
            monitoring_config=entry.monitoring_config,
            mini_wandb=context.mini_wandb,
        )

    def _build_detector_context(
        self,
        *,
        entry: SubsystemEntry,
        context: MainContext,
    ) -> DetectorContext:
        return DetectorContext(
            metric_store=context.metric_store,
            mini_wandb=context.mini_wandb,
            active_node_ids=entry.get_active_node_ids(),
            job_status=context.job_status,
        )

    def _build_recovery_context(
        self,
        *,
        entry: SubsystemEntry,
        context: MainContext,
        trigger: TriggerType,
        recovery_start_time: datetime,
    ) -> RecoveryContext:
        restart_context = RestartContext(
            node_manager=context.node_manager,
            main_job=context.main_job,
            mini_wandb=context.mini_wandb,
            notifier=context.notifier,
            on_new_run=context.on_new_run,
            actuator=entry.actuator,
            monitoring_config=entry.monitoring_config,
            has_level1_restart=entry.has_level1_restart,
        )
        return RecoveryContext(
            trigger=trigger,
            recovery_start_time=recovery_start_time,
            diagnostic_orchestrator=context.diagnostic_orchestrator,
            restart_stepper=self._restart_stepper,
            restart_context=restart_context,
            notifier=context.notifier,
            timeout_seconds=context.recovery_timeout_seconds,
            rank_pids_provider=context.rank_pids_provider,
        )

    def _should_run_detectors(
        self,
        *,
        entry: SubsystemEntry,
        context: MainContext,
    ) -> bool:
        active_nodes = entry.get_active_node_ids()
        if len(active_nodes) == 0:
            return False

        if context.tick_count <= context.registration_grace_ticks:
            return False

        return True


class RestartingMainJobStateHandler(StateHandler[RestartingMainJobState, MainContext]):
    async def step(
        self, state: RestartingMainJobState, context: MainContext
    ) -> MainState | None:
        status = await context.main_job.get_job_status()

        if status == JobStatus.RUNNING:
            fresh = context.create_fresh_subsystems()
            if state.requestor_name in fresh:
                restored = _update_externally_fulfilled(state.requestor_frozen_state)
                fresh[state.requestor_name].state_machine.force_state(restored)
            return NormalState(subsystems=fresh)

        if status == JobStatus.FAILED:
            logger.warning("main_job_restart_failed subsystem=%s", state.requestor_name)
            await safe_notify(
                context.notifier,
                title="Recovery Alert",
                content=f"Main job restart failed for subsystem '{state.requestor_name}'",
            )
            fresh = context.create_fresh_subsystems()
            return NormalState(subsystems=fresh)

        # PENDING / STOPPED — check timeout
        elapsed = (datetime.now(timezone.utc) - state.start_time).total_seconds()
        if elapsed > context.recovery_timeout_seconds:
            logger.warning(
                "main_job_restart_timeout subsystem=%s elapsed=%ds",
                state.requestor_name,
                int(elapsed),
            )
            await safe_notify(
                context.notifier,
                title="Recovery Alert",
                content=f"Main job restart for '{state.requestor_name}' timed out after {int(elapsed)}s",
            )
            fresh = context.create_fresh_subsystems()
            return NormalState(subsystems=fresh)

        return None
