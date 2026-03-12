from __future__ import annotations

from datetime import datetime

from miles.utils.ft.controller.detectors.base import DetectorContext
from miles.utils.ft.controller.state_machines.main.models import MainContext
from miles.utils.ft.controller.state_machines.recovery.models import RecoveryContext
from miles.utils.ft.controller.state_machines.restart.models import RestartContext
from miles.utils.ft.controller.state_machines.subsystem.models import SubsystemContext
from miles.utils.ft.controller.subsystem_hub import SubsystemConfig
from miles.utils.ft.controller.types import TriggerType
from miles.utils.ft.utils.state_machine import StateMachineStepper


def build_subsystem_context(
    *,
    config: SubsystemConfig,
    context: MainContext,
    recovery_stepper: StateMachineStepper,
    restart_stepper: StateMachineStepper,
) -> SubsystemContext:
    active_node_ids = config.get_active_node_ids()
    should_run = _should_run_detectors(active_node_ids=active_node_ids, context=context)
    detector_ctx = _build_detector_context(active_node_ids=active_node_ids, context=context) if should_run else None

    return SubsystemContext(
        job_status=context.job_status,
        tick_count=context.tick_count,
        should_run_detectors=should_run,
        detector_context=detector_ctx,
        notifier=context.notifier,
        detectors=config.detectors,
        cooldown=context.cooldown,
        detector_crash_tracker=context.detector_crash_tracker,
        recovery_stepper=recovery_stepper,
        recovery_context_factory=lambda trigger, start_time: _build_recovery_context(
            config=config,
            context=context,
            trigger=trigger,
            recovery_start_time=start_time,
            restart_stepper=restart_stepper,
        ),
        on_recovery_duration=context.on_recovery_duration,
        max_simultaneous_bad_nodes=context.max_simultaneous_bad_nodes,
        monitoring_config=config.monitoring_config,
        metric_store=context.metric_store,
    )


def _should_run_detectors(
    *,
    active_node_ids: set[str],
    context: MainContext,
) -> bool:
    if len(active_node_ids) == 0:
        return False

    ticks_since_run_start = context.tick_count - context.run_start_tick
    if ticks_since_run_start <= context.registration_grace_ticks:
        return False

    return True


def _build_detector_context(
    *,
    active_node_ids: set[str],
    context: MainContext,
) -> DetectorContext:
    return DetectorContext(
        metric_store=context.metric_store,
        active_node_ids=active_node_ids,
        job_status=context.job_status,
    )


def _build_recovery_context(
    *,
    config: SubsystemConfig,
    context: MainContext,
    trigger: TriggerType,
    recovery_start_time: datetime,
    restart_stepper: StateMachineStepper,
) -> RecoveryContext:
    restart_context = RestartContext(
        node_manager=context.node_manager,
        main_job=context.main_job,
        metric_store=context.metric_store,
        notifier=context.notifier,
        on_main_job_new_run=context.on_main_job_new_run,
        node_metadata=context.node_metadata,
        actuator=config.actuator,
        monitoring_config=config.monitoring_config,
        restart_mode=config.restart_mode,
    )
    return RecoveryContext(
        trigger=trigger,
        recovery_start_time=recovery_start_time,
        diagnostic_orchestrator=context.diagnostic_orchestrator,
        restart_stepper=restart_stepper,
        restart_context=restart_context,
        notifier=context.notifier,
        timeout_seconds=context.recovery_timeout_seconds,
        rank_pids_provider=context.rank_pids_provider,
    )
