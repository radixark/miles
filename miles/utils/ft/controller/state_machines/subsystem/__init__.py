from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb as _MiniWandb
from miles.utils.ft.controller.state_machines.subsystem.handlers import DetectingAnomalyHandler, RecoveringHandler
from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomalySt,
    RecoveringSt,
    SubsystemContext,
    SubsystemState,
)
from miles.utils.ft.controller.state_machines.subsystem.utils import handle_notify_human
from miles.utils.ft.utils.state_machine import StateMachineStepper

SubsystemContext.model_rebuild(_types_namespace={"MiniWandb": _MiniWandb})

_SUBSYSTEM_HANDLER_MAP: dict[type, type] = {
    DetectingAnomalySt: DetectingAnomalyHandler,
    RecoveringSt: RecoveringHandler,
}


def create_subsystem_stepper() -> StateMachineStepper:
    return StateMachineStepper(handler_map=_SUBSYSTEM_HANDLER_MAP)


__all__ = [
    "DetectingAnomalySt",
    "DetectingAnomalyHandler",
    "SubsystemContext",
    "SubsystemState",
    "RecoveringSt",
    "RecoveringHandler",
    "create_subsystem_stepper",
    "handle_notify_human",
]
