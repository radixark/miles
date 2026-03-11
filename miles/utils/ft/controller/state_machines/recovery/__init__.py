from miles.utils.ft.controller.state_machines.recovery.handlers import (
    EvictingAndRestartingHandler,
    NotifyHumansHandler,
    RealtimeChecksHandler,
    StopTimeDiagnosticsHandler,
    recovery_timeout_check,
)
from miles.utils.ft.controller.state_machines.recovery.models import (
    RECOVERY_STATE_TO_INT,
    EvictingAndRestarting,
    NotifyHumans,
    RealtimeChecks,
    RecoveryContext,
    RecoveryDone,
    RecoveryEscalated,
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
        terminal_states=frozenset({RecoveryDone, RecoveryEscalated}),
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
    "RecoveryEscalated",
    "RecoveryState",
    "StopTimeDiagnostics",
    "StopTimeDiagnosticsHandler",
    "create_recovery_stepper",
    "recovery_timeout_check",
]
