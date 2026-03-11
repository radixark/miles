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
from miles.utils.ft.controller.state_machines.subsystem import create_subsystem_stepper
from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomaly,
    SubsystemContext,
    SubsystemState,
    Recovering,
)
from miles.utils.ft.controller.state_machines.recovery.models import (
    EvictingAndRestarting,
    RecoveryContext,
)
from miles.utils.ft.controller.state_machines.restart.models import (
    ExternalExecutionResult,
    RestartContext,
    RestartingMainJob as RestartingMainJobRestart,
)
from miles.utils.ft.controller.state_machines.recovery import create_recovery_stepper
from miles.utils.ft.controller.state_machines.restart import create_restart_stepper
from miles.utils.ft.controller.subsystem import SubsystemConfig
from miles.utils.ft.controller.types import TriggerType
from miles.utils.ft.utils.state_machine import StateHandler

logger = logging.getLogger(__name__)


def _find_restart_requestor(subsystems: dict[str, SubsystemState]) -> str | None:
    for name, sub_state in subsystems.items():
        match sub_state:
            case Recovering(
                recovery=EvictingAndRestarting(
                    restart=RestartingMainJobRestart(external_execution_result=None)
                )
            ):
                return name
    return None


def _update_external_execution_result(
    frozen_state: SubsystemState,
    result: ExternalExecutionResult,
) -> SubsystemState:
    match frozen_state:
        case Recovering(
            recovery=EvictingAndRestarting(
                restart=RestartingMainJobRestart() as restart
            ) as recovery
        ):
            return frozen_state.model_copy(update={"recovery":
                recovery.model_copy(update={"restart":
                    restart.model_copy(update={"external_execution_result": result})
                })
            })
        case _:
            raise AssertionError(f"Unexpected state for _update_external_execution_result: {frozen_state}")


def _build_fresh_subsystem_states(configs: dict[str, SubsystemConfig]) -> dict[str, SubsystemState]:
    return {name: DetectingAnomaly() for name in configs}


class NormalStateHandler(StateHandler[NormalState, MainContext]):
    def __init__(self) -> None:
        self._restart_stepper = create_restart_stepper()
        self._recovery_stepper = create_recovery_stepper()
        self._subsystem_stepper = create_subsystem_stepper()

    async def step(self, state: NormalState, context: MainContext):  # type: ignore[override]
        # Step 1: Step all subsystems (loop stepper until no more transitions)
        curr_state = state
        for name, sub_state in state.subsystems.items():
            config = context.subsystem_configs[name]
            sub_ctx = self._build_subsystem_context(config=config, context=context)
            current = sub_state
            while True:
                next_state = await self._subsystem_stepper.step_once(current, sub_ctx)
                if next_state is None:
                    break
                current = next_state
            if current != sub_state:
                curr_state = NormalState(subsystems={**curr_state.subsystems, name: current})
                yield curr_state

        # Step 2: Check for RestartingMainJob(externally_fulfilled=False)
        requestor = _find_restart_requestor(curr_state.subsystems)
        if requestor is not None:
            frozen_state = curr_state.subsystems[requestor]
            logger.info("sub-SM %r requested main job restart (peek-and-freeze)", requestor)
            await context.main_job.stop_job()
            run_id = await context.main_job.submit_job()
            if context.on_new_run is not None:
                context.on_new_run(run_id)
            yield RestartingMainJobState(
                requestor_name=requestor,
                start_time=datetime.now(timezone.utc),
                requestor_frozen_state=frozen_state,
            )

    def _build_subsystem_context(
        self,
        *,
        config: SubsystemConfig,
        context: MainContext,
    ) -> SubsystemContext:
        should_run = self._should_run_detectors(config=config, context=context)
        detector_ctx = self._build_detector_context(config=config, context=context) if should_run else None

        return SubsystemContext(
            job_status=context.job_status,
            tick_count=context.tick_count,
            should_run_detectors=should_run,
            detector_context=detector_ctx,
            notifier=context.notifier,
            detectors=config.detectors,
            cooldown=context.cooldown,
            detector_crash_tracker=context.detector_crash_tracker,
            recovery_stepper=self._recovery_stepper,
            recovery_context_factory=lambda trigger, start_time: self._build_recovery_context(
                config=config,
                context=context,
                trigger=trigger,
                recovery_start_time=start_time,
            ),
            on_recovery_duration=context.on_recovery_duration,
            max_simultaneous_bad_nodes=context.max_simultaneous_bad_nodes,
            monitoring_config=config.monitoring_config,
            mini_wandb=context.mini_wandb,
        )

    def _build_detector_context(
        self,
        *,
        config: SubsystemConfig,
        context: MainContext,
    ) -> DetectorContext:
        return DetectorContext(
            metric_store=context.metric_store,
            mini_wandb=context.mini_wandb,
            active_node_ids=config.get_active_node_ids(),
            job_status=context.job_status,
        )

    def _build_recovery_context(
        self,
        *,
        config: SubsystemConfig,
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
            actuator=config.actuator,
            monitoring_config=config.monitoring_config,
            restart_mode=config.restart_mode,
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
        config: SubsystemConfig,
        context: MainContext,
    ) -> bool:
        active_nodes = config.get_active_node_ids()
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
        execution_result: ExternalExecutionResult | None = None

        if status == JobStatus.RUNNING:
            execution_result = ExternalExecutionResult.SUCCEEDED
        elif status == JobStatus.FAILED:
            execution_result = ExternalExecutionResult.FAILED
        else:
            elapsed = (datetime.now(timezone.utc) - state.start_time).total_seconds()
            if elapsed > context.recovery_timeout_seconds:
                execution_result = ExternalExecutionResult.TIMEOUT

        if execution_result is None:
            return None

        fresh_states = _build_fresh_subsystem_states(context.subsystem_configs)
        if state.requestor_name in fresh_states:
            restored = _update_external_execution_result(state.requestor_frozen_state, execution_result)
            fresh_states[state.requestor_name] = restored
        return NormalState(subsystems=fresh_states)
