from miles.utils.ft.controller.state_machines.restart.handlers import (
    EvictingHandler,
    MonitoringProgressHandler,
    StoppingAndRestartingHandler,
    WaitingForNewNodeHandler,
    iteration_progress,
)
from miles.utils.ft.controller.state_machines.restart.models import (
    Evicting,
    MonitoringProgress,
    RestartContext,
    RestartDone,
    RestartFailed,
    RestartState,
    StoppingAndRestarting,
    WaitingForNewNode,
)
from miles.utils.ft.utils.state_machine import StateMachineStepper

RestartContext.model_rebuild()

_RESTART_HANDLER_MAP: dict[type, type] = {
    Evicting: EvictingHandler,
    WaitingForNewNode: WaitingForNewNodeHandler,
    StoppingAndRestarting: StoppingAndRestartingHandler,
    MonitoringProgress: MonitoringProgressHandler,
}


def create_restart_stepper() -> StateMachineStepper:
    return StateMachineStepper(
        handler_map=_RESTART_HANDLER_MAP,
        terminal_states=frozenset({RestartDone, RestartFailed}),
    )


__all__ = [
    "Evicting",
    "EvictingHandler",
    "MonitoringProgress",
    "MonitoringProgressHandler",
    "RestartContext",
    "RestartDone",
    "RestartFailed",
    "RestartState",
    "StoppingAndRestarting",
    "StoppingAndRestartingHandler",
    "WaitingForNewNode",
    "WaitingForNewNodeHandler",
    "create_restart_stepper",
    "iteration_progress",
]
