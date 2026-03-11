from miles.utils.ft.controller.state_machines.main.context import MainContext
from miles.utils.ft.controller.state_machines.main.handlers import (
    NormalStateHandler,
    RestartingMainJobStateHandler,
)
from miles.utils.ft.controller.state_machines.main.models import (
    MainState,
    NormalState,
    RestartingMainJobState,
)
from miles.utils.ft.utils.state_machine import StateMachineStepper

_MAIN_HANDLER_MAP: dict[type, type] = {
    NormalState: NormalStateHandler,
    RestartingMainJobState: RestartingMainJobStateHandler,
}


def create_main_stepper() -> StateMachineStepper[MainState, MainContext]:
    return StateMachineStepper(handler_map=_MAIN_HANDLER_MAP)


__all__ = [
    "MainContext",
    "MainState",
    "NormalState",
    "NormalStateHandler",
    "RestartingMainJobState",
    "RestartingMainJobStateHandler",
    "create_main_stepper",
]
