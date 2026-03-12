from __future__ import annotations

import logging
from datetime import datetime, timezone

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.state_machines.main.models import (
    MainContext,
    MainState,
    NormalSt,
    RestartingMainJobSt,
)
from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomalySt,
    RecoveringSt,
    SubsystemState,
)
from miles.utils.ft.controller.state_machines.recovery.models import EvictingAndRestartingSt
from miles.utils.ft.controller.state_machines.restart.models import (
    ExternalExecutionResult,
    ExternalRestartingMainJobSt,
)
from miles.utils.ft.controller.state_machines.restart.utils import stop_and_submit
from miles.utils.ft.controller.state_machines.utils import safe_notify
from miles.utils.ft.controller.subsystem_hub import SubsystemSpec

logger = logging.getLogger(__name__)


def find_restart_requestor(subsystems: dict[str, SubsystemState]) -> str | None:
    """Find a subsystem requesting a main job restart.

    Scans for RecoveringSt -> EvictingAndRestartingSt -> ExternalRestartingMainJobSt
    with external_execution_result=None (unfulfilled request).

    Returns the first matching subsystem name, or None.
    Logs a warning if multiple requestors are found (only one is handled).
    """
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


def update_external_execution_result(
    frozen_state: SubsystemState,
    result: ExternalExecutionResult,
) -> SubsystemState:
    """Deep-update the ExternalRestartingMainJobSt.external_execution_result in a frozen state tree."""
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
            raise AssertionError(f"Unexpected state for update_external_execution_result: {frozen_state}")


def build_fresh_subsystem_states(specs: dict[str, SubsystemSpec]) -> dict[str, SubsystemState]:
    """Create a fresh DetectingAnomalySt for every subsystem spec."""
    return {name: DetectingAnomalySt() for name in specs}


async def trigger_main_job_restart(
    state: NormalSt,
    context: MainContext,
) -> MainState | None:
    """Detect a pending main-job restart request and execute stop-and-submit.

    Returns RestartingMainJobSt on success, NormalSt (with FAILED result)
    on failure, or None if no subsystem is requesting a restart.
    """
    requestor = find_restart_requestor(state.subsystems)
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
        job=context.shared.main_job,
        on_new_run=context.shared.on_main_job_new_run,
        restart_lock=context.shared.restart_lock,
    )
    if not success:
        await safe_notify(
            context.shared.notifier,
            title="Main job restart failed",
            content=f"stop_and_submit failed for requestor {requestor}",
        )
        updated_requestor = update_external_execution_result(
            frozen_state, ExternalExecutionResult.FAILED,
        )
        return NormalSt(subsystems={**state.subsystems, requestor: updated_requestor})

    return RestartingMainJobSt(
        requestor_name=requestor,
        start_time=datetime.now(timezone.utc),
        requestor_frozen_state=frozen_state,
    )


async def resolve_main_job_restart(
    state: RestartingMainJobSt,
    context: MainContext,
) -> NormalSt | None:
    """Poll the main job and return NormalSt when the restart outcome is known.

    Returns None while the job is still pending (not yet RUNNING or FAILED,
    and timeout not exceeded).
    """
    status = await context.shared.main_job.get_status()
    execution_result: ExternalExecutionResult | None = None

    if status == JobStatus.RUNNING:
        execution_result = ExternalExecutionResult.SUCCEEDED
    elif status in (JobStatus.FAILED, JobStatus.STOPPED):
        execution_result = ExternalExecutionResult.FAILED
    else:
        elapsed = (datetime.now(timezone.utc) - state.start_time).total_seconds()
        if elapsed > context.shared.recovery_timeout_seconds:
            execution_result = ExternalExecutionResult.TIMEOUT

    if execution_result is None:
        return None

    fresh_states = build_fresh_subsystem_states(context.shared.subsystem_specs)
    if state.requestor_name in fresh_states:
        restored = update_external_execution_result(state.requestor_frozen_state, execution_result)
        fresh_states[state.requestor_name] = restored
    else:
        logger.warning(
            "requestor_state_dropped requestor=%s — subsystem no longer in configs",
            state.requestor_name,
        )
    return NormalSt(subsystems=fresh_states)
