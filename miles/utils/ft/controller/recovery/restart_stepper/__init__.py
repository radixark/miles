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

RESTART_HANDLER_MAP: dict[type, type] = {
    Evicting: EvictingHandler,
    StoppingAndRestarting: StoppingAndRestartingHandler,
    MonitoringProgress: MonitoringProgressHandler,
    RestartDone: TerminalHandler,
    RestartFailed: TerminalHandler,
}

__all__ = [
    "Evicting",
    "EvictingHandler",
    "MonitoringProgress",
    "MonitoringProgressHandler",
    "RESTART_HANDLER_MAP",
    "RestartContext",
    "RestartDone",
    "RestartFailed",
    "RestartState",
    "StoppingAndRestarting",
    "StoppingAndRestartingHandler",
    "TerminalHandler",
    "iteration_progress",
]
