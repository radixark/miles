from datetime import datetime, timedelta
from enum import Enum
from typing import Literal, NamedTuple, Protocol

from pydantic import BaseModel, ConfigDict, Field


class StepValue(NamedTuple):
    step: int
    value: float


class TimedStepValue(NamedTuple):
    step: int
    timestamp: datetime
    value: float


class TrainingMetricStoreProtocol(Protocol):
    def latest(self, metric_name: str) -> float | None: ...

    def query_last_n_steps(
        self, metric_name: str, last_n: int,
    ) -> list[StepValue]: ...

    def query_time_window(
        self, metric_name: str, window: timedelta,
    ) -> list[TimedStepValue]: ...


class FtBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricSample(FtBaseModel):
    name: str
    labels: dict[str, str]
    value: float
    metric_type: Literal["gauge", "counter"] = "gauge"


class CollectorOutput(FtBaseModel):
    metrics: list[MetricSample]


class ActionType(str, Enum):
    NONE = "none"
    MARK_BAD_AND_RESTART = "mark_bad_and_restart"
    ENTER_RECOVERY = "enter_recovery"
    NOTIFY_HUMAN = "notify_human"


class TriggerType(str, Enum):
    NONE = ""
    HANG = "hang"
    NAN_LOSS = "nan_loss"
    CRASH = "crash"


class NodeFault(FtBaseModel):
    node_id: str
    reason: str


def unique_node_ids(faults: list["NodeFault"]) -> list[str]:
    """Return deduplicated node IDs from faults, preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for fault in faults:
        if fault.node_id not in seen:
            seen.add(fault.node_id)
            result.append(fault.node_id)
    return result


class Decision(FtBaseModel):
    action: ActionType
    bad_node_ids: list[str] = Field(default_factory=list)
    reason: str
    trigger: TriggerType = TriggerType.NONE

    @classmethod
    def from_node_faults(
        cls,
        faults: "list[NodeFault]",
        *,
        fallback_reason: str,
    ) -> "Decision":
        if not faults:
            return cls(action=ActionType.NONE, reason=fallback_reason)

        return cls(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=sorted(unique_node_ids(faults)),
            reason="; ".join(f.reason for f in faults),
        )


class DiagnosticResult(FtBaseModel):
    diagnostic_type: str
    node_id: str
    passed: bool
    details: str

    @classmethod
    def pass_result(
        cls, *, diagnostic_type: str, node_id: str, details: str,
    ) -> "DiagnosticResult":
        return cls(diagnostic_type=diagnostic_type, node_id=node_id, passed=True, details=details)

    @classmethod
    def fail_result(
        cls, *, diagnostic_type: str, node_id: str, details: str,
    ) -> "DiagnosticResult":
        return cls(diagnostic_type=diagnostic_type, node_id=node_id, passed=False, details=details)


class UnknownDiagnosticError(Exception):
    """Raised when a node agent is asked to run a diagnostic type it does not have."""


class NodeAgentProtocol(Protocol):
    async def run_diagnostic(
        self, diagnostic_type: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult: ...


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


class ControllerStatus(FtBaseModel):
    mode: ControllerMode
    recovery_phase: RecoveryPhase | None
    phase_history: list[RecoveryPhase] | None
    tick_count: int
    active_run_id: str | None
    bad_nodes: list[str]


FT_CONTROLLER_ACTOR_NAME: str = "ft_controller"  # deprecated: use ft_controller_actor_name()


def ft_controller_actor_name(ft_id: str) -> str:
    if not ft_id:
        return "ft_controller"
    return f"ft_controller_{ft_id}"

RECOVERY_PHASE_TO_INT: dict[RecoveryPhase, int] = {
    RecoveryPhase.CHECK_ALERTS: 1,
    RecoveryPhase.REATTEMPTING: 2,
    RecoveryPhase.MONITORING: 3,
    RecoveryPhase.DIAGNOSING: 4,
    RecoveryPhase.EVICT_AND_RESTART: 5,
    RecoveryPhase.NOTIFY: 6,
    RecoveryPhase.DONE: 7,
}
