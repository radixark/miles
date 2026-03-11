from miles.utils.ft.controller.state_machines.restart.handlers import (
    EvictingHandler,
    MonitoringProgressHandler,
    RestartingMainJobHandler,
    StoppingAndRestartingHandler,
    iteration_progress,
)
from miles.utils.ft.controller.state_machines.restart.models import (
    Evicting,
    ExternalExecutionResult,
    MonitoringProgress,
    RestartContext,
    RestartDone,
    RestartFailed,
    RestartingMainJob,
    RestartState,
    StoppingAndRestarting,
)
from miles.utils.ft.utils.state_machine import StateMachineStepper

RestartContext.model_rebuild()

_RESTART_HANDLER_MAP: dict[type, type] = {
    Evicting: EvictingHandler,
    StoppingAndRestarting: StoppingAndRestartingHandler,
    MonitoringProgress: MonitoringProgressHandler,
    RestartingMainJob: RestartingMainJobHandler,
}


def create_restart_stepper() -> StateMachineStepper:
    return StateMachineStepper(
        handler_map=_RESTART_HANDLER_MAP,
        terminal_states=frozenset({RestartDone, RestartFailed}),
    )


__all__ = [
    "Evicting",
    "EvictingHandler",
    "ExternalExecutionResult",
    "MonitoringProgress",
    "MonitoringProgressHandler",
    "RestartContext",
    "RestartDone",
    "RestartFailed",
    "RestartingMainJob",
    "RestartingMainJobHandler",
    "RestartState",
    "StoppingAndRestarting",
    "StoppingAndRestartingHandler",
    "create_restart_stepper",
    "iteration_progress",
]
