from miles.utils.ft.controller.state_machines.subsystem.handlers import (
    DetectingAnomalyHandler,
    RecoveringHandler,
)
from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomaly,
    SubsystemContext,
    SubsystemState,
    Recovering,
)
from miles.utils.ft.controller.state_machines.subsystem.utils import get_known_bad_nodes, handle_notify_human
from miles.utils.ft.utils.state_machine import StateMachineStepper

_SUBSYSTEM_HANDLER_MAP: dict[type, type] = {
    DetectingAnomaly: DetectingAnomalyHandler,
    Recovering: RecoveringHandler,
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
    "create_subsystem_stepper",
    "get_known_bad_nodes",
    "handle_notify_human",
]
