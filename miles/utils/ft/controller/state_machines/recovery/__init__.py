from miles.utils.ft.controller.recovery.recovery_stepper.handlers import (
    EvictingAndRestartingHandler,
    NotifyHumansHandler,
    RealtimeChecksHandler,
    RecoveryContext,
    StopTimeDiagnosticsHandler,
    recovery_timeout_check,
)
from miles.utils.ft.controller.recovery.recovery_stepper.states import (
    RECOVERY_STATE_TO_INT,
    EvictingAndRestarting,
    NotifyHumans,
    RealtimeChecks,
    RecoveryDone,
    RecoveryState,
    StopTimeDiagnostics,
)
from miles.utils.ft.utils.state_machine import StateMachineStepper

RECOVERY_TIMEOUT_SECONDS: int = 1800

RecoveryContext.model_rebuild()

_RECOVERY_HANDLER_MAP: dict[type, type] = {
    RealtimeChecks: RealtimeChecksHandler,
    EvictingAndRestarting: EvictingAndRestartingHandler,
    StopTimeDiagnostics: StopTimeDiagnosticsHandler,
    NotifyHumans: NotifyHumansHandler,
}


def create_recovery_stepper() -> StateMachineStepper:
    return StateMachineStepper(
        handler_map=_RECOVERY_HANDLER_MAP,
        terminal_states=frozenset({RecoveryDone}),
        pre_dispatch=recovery_timeout_check,
    )


__all__ = [
    "EvictingAndRestarting",
    "EvictingAndRestartingHandler",
    "NotifyHumans",
    "NotifyHumansHandler",
    "RECOVERY_TIMEOUT_SECONDS",
    "RECOVERY_STATE_TO_INT",
    "RealtimeChecks",
    "RealtimeChecksHandler",
    "RecoveryContext",
    "RecoveryDone",
    "RecoveryState",
    "StopTimeDiagnostics",
    "StopTimeDiagnosticsHandler",
    "create_recovery_stepper",
    "recovery_timeout_check",
]
