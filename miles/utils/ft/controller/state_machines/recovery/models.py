from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

from pydantic import ConfigDict

from miles.utils.ft.adapters.types import NotifierProtocol
from miles.utils.ft.controller.state_machines.restart.models import (
    Evicting,
    RestartContext,
    RestartState,
    StoppingAndRestarting,
)
from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol, TriggerType
from miles.utils.ft.utils.base_model import FtBaseModel


class RecoveryState(FtBaseModel):
    model_config = ConfigDict(frozen=True)


class RealtimeChecks(RecoveryState):
    pre_identified_bad_nodes: list[str] = []


class EvictingAndRestarting(RecoveryState):
    restart: RestartState
    failed_next_state: RecoveryState

    @classmethod
    def direct_restart(cls) -> EvictingAndRestarting:
        return cls(
            restart=StoppingAndRestarting(bad_node_ids=[]),
            failed_next_state=StopTimeDiagnostics(),
        )

    @classmethod
    def evict_and_restart(cls, *, bad_node_ids: list[str]) -> EvictingAndRestarting:
        return cls(
            restart=Evicting(bad_node_ids=bad_node_ids),
            failed_next_state=StopTimeDiagnostics(),
        )

    @classmethod
    def evict_and_restart_final(cls, *, bad_node_ids: list[str]) -> EvictingAndRestarting:
        return cls(
            restart=Evicting(bad_node_ids=bad_node_ids),
            failed_next_state=NotifyHumans(state_before="EvictingAndRestarting"),
        )


class StopTimeDiagnostics(RecoveryState):
    pass


class NotifyHumans(RecoveryState):
    state_before: str


class RecoveryDone(RecoveryState):
    pass


RECOVERY_STATE_TO_INT: dict[type[RecoveryState], int] = {
    RealtimeChecks: 1,
    EvictingAndRestarting: 2,
    StopTimeDiagnostics: 4,
    NotifyHumans: 5,
    RecoveryDone: 6,
}


class RecoveryContext(FtBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # per-call
    trigger: TriggerType
    recovery_start_time: datetime

    # deps
    diagnostic_orchestrator: DiagnosticOrchestratorProtocol
    restart_stepper: Callable[[RestartState, RestartContext], Awaitable[RestartState | None]]
    restart_context: RestartContext
    notifier: NotifierProtocol | None
    timeout_seconds: int
    rank_pids_provider: Callable[[str], dict[int, int]] | None
