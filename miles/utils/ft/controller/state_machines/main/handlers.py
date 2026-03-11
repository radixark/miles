from __future__ import annotations

import logging
from datetime import datetime

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.detectors.base import DetectorContext
from miles.utils.ft.controller.state_machines.main.context import MainContext
from miles.utils.ft.controller.state_machines.main.models import (
    MainState,
    NormalState,
    RestartingMainJobState,
)
from miles.utils.ft.controller.state_machines.subsystem.models import SubsystemContext, RestartedMainJob, RestartingMainJob
from miles.utils.ft.controller.state_machines.recovery.models import RecoveryContext
from miles.utils.ft.controller.state_machines.restart.models import RestartContext
from miles.utils.ft.controller.state_machines.recovery import create_recovery_stepper
from miles.utils.ft.controller.state_machines.restart import create_restart_stepper
from miles.utils.ft.controller.state_machines.utils import safe_notify
from miles.utils.ft.controller.subsystem import SubsystemEntry
from miles.utils.ft.controller.types import TriggerType
from miles.utils.ft.utils.state_machine import StateHandler

logger = logging.getLogger(__name__)


class NormalStateHandler(StateHandler[NormalState, MainContext]):
    def __init__(self) -> None:
        self._restart_stepper = create_restart_stepper()
        self._recovery_stepper = create_recovery_stepper()

    async def step(self, state: NormalState, context: MainContext) -> MainState | None:
        # Step 1: Step all sub-SMs (skip those already requesting restart)
        for name, entry in state.subsystems.items():
            if isinstance(entry.state_machine.state, RestartingMainJob):
                continue
            sub_ctx = self._build_subsystem_context(entry=entry, context=context)
            await entry.state_machine.step(sub_ctx)

        # Step 2: Check if any sub-SM entered RestartingMainJob
        for name, entry in state.subsystems.items():
            if isinstance(entry.state_machine.state, RestartingMainJob):
                logger.info("sub-SM %r requested main job restart", name)
                await context.main_job.stop_job()
                run_id = await context.main_job.submit_job()
                if context.on_new_run is not None:
                    context.on_new_run(run_id)
                return RestartingMainJobState(requestor_name=name)

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
        has_level1 = hasattr(entry.actuator, 'stop') and not isinstance(
            entry.actuator, type
        )
        try:
            from miles.utils.ft.adapters.impl.training_actuator import TrainingSubsystemActuator
            has_level1 = not isinstance(entry.actuator, TrainingSubsystemActuator)
        except ImportError:
            has_level1 = True

        restart_context = RestartContext(
            node_manager=context.node_manager,
            main_job=context.main_job,
            mini_wandb=context.mini_wandb,
            notifier=context.notifier,
            on_new_run=context.on_new_run,
            actuator=entry.actuator,
            monitoring_config=entry.monitoring_config,
            has_level1_restart=has_level1,
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
                fresh[state.requestor_name].state_machine.force_state(RestartedMainJob())
            return NormalState(subsystems=fresh)

        if status == JobStatus.FAILED:
            logger.warning("main job restart failed, rebuilding subsystems for retry")
            fresh = context.create_fresh_subsystems()
            return NormalState(subsystems=fresh)

        # PENDING / STOPPED — keep waiting
        return None
