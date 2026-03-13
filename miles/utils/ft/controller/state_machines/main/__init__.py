from miles.utils.ft.controller.state_machines.main.handlers import NormalHandler, RestartingMainJobHandler
from miles.utils.ft.controller.state_machines.main.models import MainContext, MainState, NormalSt, RestartingMainJobSt
from miles.utils.ft.utils.state_machine import StateMachineStepper

MainContext.model_rebuild()

_MAIN_HANDLER_MAP: dict[type, type] = {
    NormalSt: NormalHandler,
    RestartingMainJobSt: RestartingMainJobHandler,
}


def create_main_stepper() -> StateMachineStepper[MainState, MainContext]:
    return StateMachineStepper(handler_map=_MAIN_HANDLER_MAP)


__all__ = [
    "MainContext",
    "MainState",
    "NormalSt",
    "NormalHandler",
    "RestartingMainJobSt",
    "RestartingMainJobHandler",
    "create_main_stepper",
]
