from miles.utils.ft.controller.state_machines.main.handlers import (
    DetectingAnomalyHandler,
    RecoveringHandler,
    RestartedMainJobHandler,
    RestartingMainJobHandler,
)
from miles.utils.ft.controller.state_machines.main.models import (
    DetectingAnomaly,
    MainContext,
    MainState,
    Recovering,
    RestartedMainJob,
    RestartingMainJob,
)
from miles.utils.ft.controller.state_machines.main.utils import get_known_bad_nodes, handle_notify_human
from miles.utils.ft.utils.state_machine import StateMachineStepper

_MAIN_HANDLER_MAP: dict[type, type] = {
    DetectingAnomaly: DetectingAnomalyHandler,
    Recovering: RecoveringHandler,
    RestartingMainJob: RestartingMainJobHandler,
    RestartedMainJob: RestartedMainJobHandler,
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
    "RestartedMainJob",
    "RestartedMainJobHandler",
    "RestartingMainJob",
    "RestartingMainJobHandler",
    "create_main_stepper",
    "get_known_bad_nodes",
    "handle_notify_human",
]
