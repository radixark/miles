from miles.utils.ft.controller.main_stepper.handlers import DetectingAnomalyHandler, RecoveringHandler
from miles.utils.ft.controller.main_stepper.states import DetectingAnomaly, MainState, Recovering
from miles.utils.ft.controller.main_stepper.utils import MainContext, get_known_bad_nodes
from miles.utils.ft.utils.state_machine import StateMachineStepper

_MAIN_HANDLER_MAP: dict[type, type] = {
    DetectingAnomaly: DetectingAnomalyHandler,
    Recovering: RecoveringHandler,
}


def create_main_stepper() -> StateMachineStepper:
    return StateMachineStepper(handler_map=_MAIN_HANDLER_MAP)


__all__ = [
    "DetectingAnomaly",
    "DetectingAnomalyHandler",
    "MainContext",
    "MainState",
    "Recovering",
    "RecoveringHandler",
    "create_main_stepper",
    "get_known_bad_nodes",
]
