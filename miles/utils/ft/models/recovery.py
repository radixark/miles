from dataclasses import dataclass
from enum import Enum

from miles.utils.ft.models.base import FtBaseModel


class RecoveryPhase(str, Enum):
    CHECK_ALERTS = "check_alerts"
    REATTEMPTING = "reattempting"
    MONITORING = "monitoring"
    DIAGNOSING = "diagnosing"
    EVICT_AND_RESTART = "evict_and_restart"
    NOTIFY = "notify"
    DONE = "done"


class ControllerMode(str, Enum):
    MONITORING = "monitoring"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class RecoverySnapshot:
    in_progress: bool
    phase: RecoveryPhase | None
    phase_history: list[RecoveryPhase] | None
    diagnosing_nodes: list[str]
    bad_nodes_confirmed: bool


class ControllerStatus(FtBaseModel):
    mode: ControllerMode
    recovery_phase: RecoveryPhase | None
    phase_history: list[RecoveryPhase] | None
    tick_count: int
    active_run_id: str | None
    bad_nodes: list[str]
    recovery_in_progress: bool
    bad_nodes_confirmed: bool
    latest_iteration: int | None


RECOVERY_PHASE_TO_INT: dict[RecoveryPhase, int] = {
    RecoveryPhase.CHECK_ALERTS: 1,
    RecoveryPhase.REATTEMPTING: 2,
    RecoveryPhase.MONITORING: 3,
    RecoveryPhase.DIAGNOSING: 4,
    RecoveryPhase.EVICT_AND_RESTART: 5,
    RecoveryPhase.NOTIFY: 6,
    RecoveryPhase.DONE: 7,
}

_BAD_NODES_CONFIRMED_PHASES: frozenset[RecoveryPhase] = frozenset({
    RecoveryPhase.EVICT_AND_RESTART,
    RecoveryPhase.NOTIFY,
    RecoveryPhase.DONE,
})
