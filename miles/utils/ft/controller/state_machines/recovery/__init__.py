from miles.utils.ft.controller.state_machines.recovery.handlers import (
    EvictingAndRestartingHandler,
    NotifyHumansHandler,
    RealtimeChecksHandler,
    StopTimeDiagnosticsHandler,
    recovery_timeout_check,
)
from miles.utils.ft.controller.state_machines.recovery.models import (
    RECOVERY_STATE_TO_INT,
    EvictingAndRestartingSt,
    NotifyHumansSt,
    RealtimeChecksSt,
    RecoveryContext,
    RecoveryDoneSt,
    RecoveryState,
    StopTimeDiagnosticsSt,
)
from miles.utils.ft.utils.state_machine import StateMachineStepper

RECOVERY_TIMEOUT_SECONDS: int = 1800

RecoveryContext.model_rebuild()

_RECOVERY_HANDLER_MAP: dict[type, type] = {
    RealtimeChecksSt: RealtimeChecksHandler,
    EvictingAndRestartingSt: EvictingAndRestartingHandler,
    StopTimeDiagnosticsSt: StopTimeDiagnosticsHandler,
    NotifyHumansSt: NotifyHumansHandler,
}


def create_recovery_stepper() -> StateMachineStepper:
    return StateMachineStepper(
        handler_map=_RECOVERY_HANDLER_MAP,
        terminal_states=frozenset({RecoveryDoneSt}),
        pre_dispatch=recovery_timeout_check,
    )


__all__ = [
    "EvictingAndRestartingSt",
    "EvictingAndRestartingHandler",
    "NotifyHumansSt",
    "NotifyHumansHandler",
    "RECOVERY_TIMEOUT_SECONDS",
    "RECOVERY_STATE_TO_INT",
    "RealtimeChecksSt",
    "RealtimeChecksHandler",
    "RecoveryContext",
    "RecoveryDoneSt",
    "RecoveryState",
    "StopTimeDiagnosticsSt",
    "StopTimeDiagnosticsHandler",
    "create_recovery_stepper",
    "recovery_timeout_check",
]
