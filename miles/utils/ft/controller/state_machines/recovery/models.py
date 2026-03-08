from __future__ import annotations

from pydantic import ConfigDict

from miles.utils.ft.controller.recovery.restart_stepper.states import Evicting, RestartState, StoppingAndRestarting
from miles.utils.ft.models.base import FtBaseModel


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
