from miles.utils.ft.controller.recovery.restart_stepper.handlers import (
    EvictingHandler,
    MonitoringProgressHandler,
    RestartContext,
    StoppingAndRestartingHandler,
    TerminalHandler,
    iteration_progress,
)
from miles.utils.ft.controller.recovery.restart_stepper.states import (
    Evicting,
    MonitoringProgress,
    RestartDone,
    RestartFailed,
    RestartState,
    StoppingAndRestarting,
)
from miles.utils.ft.utils.state_machine import StateMachineStepper

RestartContext.model_rebuild()

_RESTART_HANDLER_MAP: dict[type, type] = {
    Evicting: EvictingHandler,
    StoppingAndRestarting: StoppingAndRestartingHandler,
    MonitoringProgress: MonitoringProgressHandler,
    RestartDone: TerminalHandler,
    RestartFailed: TerminalHandler,
}


def create_restart_stepper() -> StateMachineStepper:
    return StateMachineStepper(handler_map=_RESTART_HANDLER_MAP)


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
    "TerminalHandler",
    "create_restart_stepper",
    "iteration_progress",
]
