from datetime import datetime
from enum import Enum
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict


class StepValue(NamedTuple):
    step: int
    value: float


class TimedStepValue(NamedTuple):
    step: int
    timestamp: datetime
    value: float


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
    HARDWARE = "hardware"
    NETWORK = "network"
    MFU_DECLINE = "mfu_decline"


class NodeFault(FtBaseModel):
    node_id: str
    reason: str


class Decision(FtBaseModel):
    action: ActionType
    bad_node_ids: list[str] = []
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

        seen: set[str] = set()
        bad_node_ids: list[str] = []
        for fault in faults:
            if fault.node_id not in seen:
                seen.add(fault.node_id)
                bad_node_ids.append(fault.node_id)

        return cls(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=sorted(bad_node_ids),
            reason="; ".join(f.reason for f in faults),
        )


class DiagnosticResult(FtBaseModel):
    diagnostic_type: str
    node_id: str
    passed: bool
    details: str
