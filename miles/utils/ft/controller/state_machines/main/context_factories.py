from __future__ import annotations

from datetime import datetime

from miles.utils.ft.controller.detectors.base import DetectorContext
from miles.utils.ft.controller.state_machines.main.models import MainContext
from miles.utils.ft.controller.state_machines.recovery.models import RecoveryContext
from miles.utils.ft.controller.state_machines.restart.models import RestartContext
from miles.utils.ft.controller.state_machines.subsystem.models import SubsystemContext
from miles.utils.ft.controller.subsystem_hub import RestartMode, SubsystemSpec
from miles.utils.ft.controller.types import TriggerType
from miles.utils.ft.utils.state_machine import StateMachineStepper


def build_subsystem_context(
    *,
    spec: SubsystemSpec,
    context: MainContext,
    recovery_stepper: StateMachineStepper,
    restart_stepper: StateMachineStepper,
) -> SubsystemContext:
    active_node_ids = spec.runtime.get_active_node_ids()
    should_run = _should_run_detectors(active_node_ids=active_node_ids, context=context)
    detector_ctx = _build_detector_context(active_node_ids=active_node_ids, context=context) if should_run else None

    return SubsystemContext(
        job_status=context.job_status,
        tick_count=context.tick_count,
        should_run_detectors=should_run,
        detector_context=detector_ctx,
        notifier=context.shared.notifier,
        detectors=spec.config.detectors,
        cooldown=spec.runtime.cooldown,
        detector_crash_tracker=context.shared.detector_crash_tracker,
        recovery_stepper=recovery_stepper,
        recovery_context_factory=lambda trigger, start_time: _build_recovery_context(
            spec=spec,
            context=context,
            trigger=trigger,
            recovery_start_time=start_time,
            restart_stepper=restart_stepper,
        ),
        on_recovery_duration=context.shared.on_recovery_duration,
        max_simultaneous_bad_nodes=context.shared.max_simultaneous_bad_nodes,
        monitoring_config=spec.config.monitoring_config,
        metric_store=context.shared.metric_store,
    )


def _should_run_detectors(
    *,
    active_node_ids: frozenset[str],
    context: MainContext,
) -> bool:
    if len(active_node_ids) == 0:
        return False

    ticks_since_run_start = max(0, context.tick_count - context.run_start_tick)
    if ticks_since_run_start <= context.shared.registration_grace_ticks:
        return False

    return True


def _build_detector_context(
    *,
    active_node_ids: frozenset[str],
    context: MainContext,
) -> DetectorContext:
    return DetectorContext(
        metric_store=context.shared.metric_store,
        active_node_ids=active_node_ids,
        job_status=context.job_status,
        active_run_id=context.shared.metric_store.mini_wandb.active_run_id,
    )


def _build_recovery_context(
    *,
    spec: SubsystemSpec,
    context: MainContext,
    trigger: TriggerType,
    recovery_start_time: datetime,
    restart_stepper: StateMachineStepper,
) -> RecoveryContext:
    is_main_job = spec.config.restart_mode == RestartMode.MAIN_JOB
    restart_context = RestartContext(
        node_manager=context.shared.node_manager,
        main_job=context.shared.main_job,
        metric_store=context.shared.metric_store,
        notifier=context.shared.notifier,
        on_new_run=context.shared.on_main_job_new_run if is_main_job else None,
        node_metadata=context.node_metadata,
        actuator=spec.runtime.actuator,
        monitoring_config=spec.config.monitoring_config,
        is_main_job_restart=is_main_job,
        restart_lock=context.shared.restart_lock,
        on_node_evicted=context.on_node_evicted,
    )
    return RecoveryContext(
        trigger=trigger,
        recovery_start_time=recovery_start_time,
        diagnostic_orchestrator=context.shared.diagnostic_orchestrator,
        restart_stepper=restart_stepper,
        restart_context=restart_context,
        notifier=context.shared.notifier,
        timeout_seconds=context.shared.recovery_timeout_seconds,
        rank_pids_provider=context.shared.rank_pids_provider,
    )
