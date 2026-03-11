from miles.utils.ft.controller.state_machines.subsystem.handlers import (
    DetectingAnomalyHandler,
    RecoveringHandler,
    RestartedMainJobHandler,
    RestartingMainJobHandler,
)
from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomaly,
    SubsystemContext,
    SubsystemState,
    Recovering,
    RestartedMainJob,
    RestartingMainJob,
)
from miles.utils.ft.controller.state_machines.subsystem.utils import get_known_bad_nodes, handle_notify_human
from miles.utils.ft.utils.state_machine import StateMachineStepper

_SUBSYSTEM_HANDLER_MAP: dict[type, type] = {
    DetectingAnomaly: DetectingAnomalyHandler,
    Recovering: RecoveringHandler,
    RestartingMainJob: RestartingMainJobHandler,
    RestartedMainJob: RestartedMainJobHandler,
}


def create_subsystem_stepper() -> StateMachineStepper:
    return StateMachineStepper(handler_map=_SUBSYSTEM_HANDLER_MAP)


__all__ = [
    "DetectingAnomaly",
    "DetectingAnomalyHandler",
    "SubsystemContext",
    "SubsystemState",
    "Recovering",
    "RecoveringHandler",
    "RestartedMainJob",
    "RestartedMainJobHandler",
    "RestartingMainJob",
    "RestartingMainJobHandler",
    "create_subsystem_stepper",
    "get_known_bad_nodes",
    "handle_notify_human",
]
