from miles.utils.ft.controller.state_machines.recovery.handlers import (
    EvictingAndRestartingHandler,
    NotifyHumansHandler,
    RealtimeChecksHandler,
    StopTimeDiagnosticsHandler,
    recovery_timeout_check,
)
from miles.utils.ft.controller.state_machines.recovery.models import (
    EvictingAndRestartingSt,
    NotifyHumansSt,
    RealtimeChecksSt,
    RecoveryContext,
    RecoveryDoneSt,
    RecoveryState,
    StopTimeDiagnosticsSt,
)
from miles.utils.ft.controller.state_machines.recovery.transitions import (
    direct_restart,
    evict_and_restart_final,
    evict_and_restart_next_stop_time_diag,
)
from miles.utils.ft.utils.state_machine import StateMachineStepper

RECOVERY_TIMEOUT_SECONDS: int = 3600

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
    "RealtimeChecksSt",
    "RealtimeChecksHandler",
    "RecoveryContext",
    "RecoveryDoneSt",
    "RecoveryState",
    "StopTimeDiagnosticsSt",
    "StopTimeDiagnosticsHandler",
    "create_recovery_stepper",
    "direct_restart",
    "evict_and_restart_final",
    "evict_and_restart_next_stop_time_diag",
    "recovery_timeout_check",
]
