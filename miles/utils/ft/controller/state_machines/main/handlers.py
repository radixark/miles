from __future__ import annotations

import logging
from datetime import datetime, timezone

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.state_machines.main.context_factories import build_subsystem_context
from miles.utils.ft.controller.state_machines.main.models import (
    MainContext,
    MainState,
    NormalState,
    RestartingMainJobState,
)
from miles.utils.ft.controller.state_machines.subsystem import create_subsystem_stepper
from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomaly,
    SubsystemState,
    Recovering,
)
from miles.utils.ft.controller.state_machines.recovery.models import (
    EvictingAndRestarting,
)
from miles.utils.ft.controller.state_machines.restart.models import (
    ExternalExecutionResult,
    RestartingMainJob as RestartingMainJobRestart,
)
from miles.utils.ft.controller.state_machines.recovery import create_recovery_stepper
from miles.utils.ft.controller.state_machines.restart import create_restart_stepper
from miles.utils.ft.controller.subsystem import SubsystemConfig
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
            sub_ctx = build_subsystem_context(
                config=config,
                context=context,
                recovery_stepper=self._recovery_stepper,
                restart_stepper=self._restart_stepper,
            )
            current = sub_state
            while True:
                previous = current
                async for next_state in self._subsystem_stepper(current, sub_ctx):
                    current = next_state
                if current is previous or current == previous:
                    break
            if current != sub_state:
                curr_state = NormalState(subsystems={**curr_state.subsystems, name: current})
                yield curr_state

        # Step 2: Check for RestartingMainJob(external_execution_result=None)
        escalation = await self._check_main_job_restart(curr_state, context)
        if escalation is not None:
            yield escalation

    async def _check_main_job_restart(
        self,
        state: NormalState,
        context: MainContext,
    ) -> RestartingMainJobState | None:
        requestor = _find_restart_requestor(state.subsystems)
        if requestor is None:
            return None

        frozen_state = state.subsystems[requestor]
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
