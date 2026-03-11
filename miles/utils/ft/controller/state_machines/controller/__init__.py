from miles.utils.ft.controller.state_machines.controller.context import ControllerContext
from miles.utils.ft.controller.state_machines.controller.handlers import (
    NormalStateHandler,
    RestartingMainJobStateHandler,
)
from miles.utils.ft.controller.state_machines.controller.models import (
    ControllerState,
    NormalState,
    RestartingMainJobState,
)
from miles.utils.ft.utils.state_machine import StateMachineStepper

_CONTROLLER_HANDLER_MAP: dict[type, type] = {
    NormalState: NormalStateHandler,
    RestartingMainJobState: RestartingMainJobStateHandler,
}


def create_controller_stepper() -> StateMachineStepper[ControllerState, ControllerContext]:
    return StateMachineStepper(handler_map=_CONTROLLER_HANDLER_MAP)


__all__ = [
    "ControllerContext",
    "ControllerState",
    "NormalState",
    "NormalStateHandler",
    "RestartingMainJobState",
    "RestartingMainJobStateHandler",
    "create_controller_stepper",
]
