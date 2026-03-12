from __future__ import annotations

import logging
from datetime import datetime, timezone

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.state_machines.main.context_factories import build_subsystem_context
from miles.utils.ft.controller.state_machines.main.models import (
    MainContext,
    MainState,
    NormalSt,
    RestartingMainJobSt,
)
from miles.utils.ft.controller.state_machines.subsystem import create_subsystem_stepper
from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomalySt,
    SubsystemState,
    RecoveringSt,
)
from miles.utils.ft.controller.state_machines.recovery.models import (
    EvictingAndRestartingSt,
)
from miles.utils.ft.controller.state_machines.restart.models import (
    ExternalExecutionResult,
    ExternalRestartingMainJobSt,
)
from miles.utils.ft.controller.state_machines.recovery import create_recovery_stepper
from miles.utils.ft.controller.state_machines.restart import create_restart_stepper
from miles.utils.ft.controller.state_machines.restart.utils import stop_and_submit
from miles.utils.ft.controller.state_machines.utils import safe_notify
from miles.utils.ft.controller.subsystem_hub import RestartMode, SubsystemSpec
from miles.utils.ft.utils.state_machine import StateHandler, run_stepper_to_convergence

logger = logging.getLogger(__name__)


def _find_restart_requestor(subsystems: dict[str, SubsystemState]) -> str | None:
    requestor: str | None = None
    for name, sub_state in subsystems.items():
        match sub_state:
            case RecoveringSt(
                recovery=EvictingAndRestartingSt(
                    restart=ExternalRestartingMainJobSt(external_execution_result=None)
                )
            ):
                if requestor is None:
                    requestor = name
                else:
                    # Intentional design choice (not a bug): concurrent MAIN_JOB
                    # restart requestors are expected to be extremely rare in
                    # production, so we only handle one requestor here.
                    # Unless product requirements change, audits should not flag
                    # this single-requestor behavior as a standalone issue.
                    logger.warning(
                        "multiple_restart_requestors found=%s handled=%s",
                        name,
                        requestor,
                    )
    return requestor


def _update_external_execution_result(
    frozen_state: SubsystemState,
    result: ExternalExecutionResult,
) -> SubsystemState:
    match frozen_state:
        case RecoveringSt(
            recovery=EvictingAndRestartingSt(
                restart=ExternalRestartingMainJobSt() as restart
            ) as recovery
        ):
            return frozen_state.model_copy(update={"recovery":
                recovery.model_copy(update={"restart":
                    restart.model_copy(update={"external_execution_result": result})
                })
            })
        case _:
            raise AssertionError(f"Unexpected state for _update_external_execution_result: {frozen_state}")


def _build_fresh_subsystem_states(specs: dict[str, SubsystemSpec]) -> dict[str, SubsystemState]:
    return {name: DetectingAnomalySt() for name in specs}


class NormalHandler(StateHandler[NormalSt, MainContext]):
    def __init__(self) -> None:
        self._restart_stepper = create_restart_stepper()
        self._recovery_stepper = create_recovery_stepper()
        self._subsystem_stepper = create_subsystem_stepper()

    async def step(self, state: NormalSt, context: MainContext):  # type: ignore[override]
        assert set(state.subsystems.keys()) == set(context.subsystem_specs.keys()), (
            f"subsystem keys out of sync: state={set(state.subsystems.keys())} "
            f"configs={set(context.subsystem_specs.keys())}"
        )

        # Step 1: Step all subsystems to convergence
        curr_state = state
        del state

        for name in sorted(curr_state.subsystems):
            sub_ctx = build_subsystem_context(
                spec=context.subsystem_specs[name],
                context=context,
                recovery_stepper=self._recovery_stepper,
                restart_stepper=self._restart_stepper,
            )
            old_sub_state = curr_state.subsystems[name]
            async for new_sub_state in run_stepper_to_convergence(self._subsystem_stepper, old_sub_state, sub_ctx):
                curr_state = NormalSt(subsystems={**curr_state.subsystems, name: new_sub_state})
                yield curr_state

        # Step 2: Check for RestartingMainJob(external_execution_result=None)
        if (s := await self._check_main_job_restart(curr_state, context)) is not None:
            yield s

    async def _check_main_job_restart(
        self,
        state: NormalSt,
        context: MainContext,
    ) -> RestartingMainJobSt | None:
        requestor = _find_restart_requestor(state.subsystems)
        if requestor is None:
            return None

        for name, sub_state in state.subsystems.items():
            if name != requestor and isinstance(sub_state, RecoveringSt):
                # Intentional trade-off: to keep the MAIN_JOB restart path simple,
                # non-requestor recovery progress is dropped and later re-detected.
                # This is acceptable under the assumption that multiple concurrent
                # requestors almost never happen in normal operation.
                logger.warning(
                    "subsystem_recovery_discarded name=%s phase=%s",
                    name,
                    type(sub_state.recovery).__name__,
                )

        frozen_state = state.subsystems[requestor]

        logger.info("sub-SM %r requested main job restart (peek-and-freeze)", requestor)
        success = await stop_and_submit(
            job=context.main_job,
            on_main_job_new_run=context.on_main_job_new_run,
            restart_mode=RestartMode.MAIN_JOB,
        )
        if not success:
            await safe_notify(
                context.notifier,
                title="Main job restart failed",
                content=f"stop_and_submit failed for requestor {requestor}",
            )
            return None

        return RestartingMainJobSt(
            requestor_name=requestor,
            start_time=datetime.now(timezone.utc),
            requestor_frozen_state=frozen_state,
        )


class RestartingMainJobHandler(StateHandler[RestartingMainJobSt, MainContext]):
    async def step(
        self, state: RestartingMainJobSt, context: MainContext
    ) -> MainState | None:
        status = await context.main_job.get_status()
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

        fresh_states = _build_fresh_subsystem_states(context.subsystem_specs)
        if state.requestor_name in fresh_states:
            restored = _update_external_execution_result(state.requestor_frozen_state, execution_result)
            fresh_states[state.requestor_name] = restored
        else:
            logger.warning(
                "requestor_state_dropped requestor=%s — subsystem no longer in configs",
                state.requestor_name,
            )
        return NormalSt(subsystems=fresh_states)
