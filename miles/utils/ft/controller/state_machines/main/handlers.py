from __future__ import annotations

from miles.utils.ft.controller.state_machines.main.models import MainContext, MainState, NormalSt, RestartingMainJobSt
from miles.utils.ft.controller.state_machines.main.restart_coordinator import (
    build_fresh_subsystem_states,
    find_restart_requestor,
    resolve_main_job_restart,
    trigger_main_job_restart,
    update_external_execution_result,
)
from miles.utils.ft.controller.state_machines.main.subsystem_runner import advance_subsystems
from miles.utils.ft.controller.state_machines.recovery import create_recovery_stepper
from miles.utils.ft.controller.state_machines.restart import create_restart_stepper
from miles.utils.ft.controller.state_machines.subsystem import create_subsystem_stepper
from miles.utils.ft.utils.state_machine import StateHandler

# Re-export for backward compatibility with existing test imports
_find_restart_requestor = find_restart_requestor
_update_external_execution_result = update_external_execution_result
_build_fresh_subsystem_states = build_fresh_subsystem_states


class NormalHandler(StateHandler[NormalSt, MainContext]):
    def __init__(self) -> None:
        self._restart_stepper = create_restart_stepper()
        self._recovery_stepper = create_recovery_stepper()
        self._subsystem_stepper = create_subsystem_stepper()

    async def step(self, state: NormalSt, context: MainContext):  # type: ignore[override]
        # Production invariant: subsystem topology is static after controller
        # bootstrap, so state keys and config keys must always match.
        # Keep assert here to fail fast on programmer/config wiring errors.
        assert set(state.subsystems.keys()) == set(context.shared.subsystem_specs.keys()), (
            f"subsystem keys out of sync: state={set(state.subsystems.keys())} "
            f"configs={set(context.shared.subsystem_specs.keys())}"
        )

        # Step 1: Step all subsystems to convergence
        current_state = state
        del state

        async for next_state in advance_subsystems(
            current_state,
            context,
            subsystem_stepper=self._subsystem_stepper,
            recovery_stepper=self._recovery_stepper,
            restart_stepper=self._restart_stepper,
            on_convergence_failure=context.shared.on_convergence_failure,
        ):
            current_state = next_state
            yield current_state

        # Step 2: Check for RestartingMainJob(external_execution_result=None)
        if (s := await trigger_main_job_restart(current_state, context)) is not None:
            yield s


class RestartingMainJobHandler(StateHandler[RestartingMainJobSt, MainContext]):
    async def step(self, state: RestartingMainJobSt, context: MainContext) -> MainState | None:
        return await resolve_main_job_restart(state=state, context=context)
