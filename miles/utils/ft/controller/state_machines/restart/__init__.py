from miles.utils.ft.controller.state_machines.restart.handlers import (
    EvictingHandler,
    MonitoringProgressHandler,
    ExternalRestartingMainJobHandler,
    StoppingAndRestartingHandler,
    iteration_progress,
)
from miles.utils.ft.controller.state_machines.restart.models import (
    EvictingSt,
    ExternalExecutionResult,
    MonitoringConfig,
    MonitoringIterationProgressConfig,
    MonitoringProgressSt,
    MonitoringSustainedAliveConfig,
    RestartContext,
    RestartDoneSt,
    RestartFailedSt,
    ExternalRestartingMainJobSt,
    RestartState,
    StoppingAndRestartingSt,
)
from miles.utils.ft.utils.state_machine import StateMachineStepper

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb as _MiniWandb

RestartContext.model_rebuild(_types_namespace={"MiniWandb": _MiniWandb})

_RESTART_HANDLER_MAP: dict[type, type] = {
    EvictingSt: EvictingHandler,
    StoppingAndRestartingSt: StoppingAndRestartingHandler,
    MonitoringProgressSt: MonitoringProgressHandler,
    ExternalRestartingMainJobSt: ExternalRestartingMainJobHandler,
}


def create_restart_stepper() -> StateMachineStepper:
    return StateMachineStepper(
        handler_map=_RESTART_HANDLER_MAP,
        terminal_states=frozenset({RestartDoneSt, RestartFailedSt}),
    )


__all__ = [
    "EvictingSt",
    "EvictingHandler",
    "ExternalExecutionResult",
    "MonitoringConfig",
    "MonitoringIterationProgressConfig",
    "MonitoringProgressSt",
    "MonitoringProgressHandler",
    "MonitoringSustainedAliveConfig",
    "RestartContext",
    "RestartDoneSt",
    "RestartFailedSt",
    "ExternalRestartingMainJobSt",
    "ExternalRestartingMainJobHandler",
    "RestartState",
    "StoppingAndRestartingSt",
    "StoppingAndRestartingHandler",
    "create_restart_stepper",
    "iteration_progress",
]
