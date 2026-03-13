from __future__ import annotations

from collections.abc import AsyncGenerator, Callable

from miles.utils.ft.controller.state_machines.main.context_factories import build_subsystem_context
from miles.utils.ft.controller.state_machines.main.models import MainContext, NormalSt
from miles.utils.ft.controller.state_machines.main.restart_coordinator import has_pending_main_job_restart
from miles.utils.ft.controller.state_machines.subsystem.models import SubsystemState
from miles.utils.ft.utils.state_machine import StateMachineStepper, run_stepper_to_convergence


async def advance_subsystems(
    state: NormalSt,
    context: MainContext,
    *,
    subsystem_stepper: StateMachineStepper,
    recovery_stepper: StateMachineStepper,
    restart_stepper: StateMachineStepper,
    on_convergence_failure: Callable[[], None] | None,
) -> AsyncGenerator[NormalSt, None]:
    """Step every subsystem to convergence, yielding each intermediate state."""
    curr_state = state

    for name in sorted(curr_state.subsystems):

        def _context_factory(_current_sub_state: SubsystemState, _name: str = name) -> object:
            return build_subsystem_context(
                spec=context.shared.subsystem_specs[_name],
                context=context,
                recovery_stepper=recovery_stepper,
                restart_stepper=restart_stepper,
            )

        old_sub_state = curr_state.subsystems[name]
        async for new_sub_state in run_stepper_to_convergence(
            subsystem_stepper,
            old_sub_state,
            context_factory=_context_factory,
            on_convergence_failure=on_convergence_failure,
        ):
            curr_state = NormalSt(subsystems={**curr_state.subsystems, name: new_sub_state})
            yield curr_state

        # Skip remaining subsystems — trigger_main_job_restart() will discard their work anyway
        if has_pending_main_job_restart(curr_state.subsystems):
            return
