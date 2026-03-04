from enum import Enum

from pydantic import BaseModel, ConfigDict


class FtBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricSample(FtBaseModel):
    name: str
    labels: dict[str, str]
    value: float


class CollectorOutput(FtBaseModel):
    metrics: list[MetricSample]


class ActionType(str, Enum):
    NONE = "none"
    MARK_BAD_AND_RESTART = "mark_bad_and_restart"
    ENTER_RECOVERY = "enter_recovery"
    NOTIFY_HUMAN = "notify_human"


class Decision(FtBaseModel):
    action: ActionType
    bad_node_ids: list[str] = []
    reason: str
    trigger: str = ""


class DiagnosticResult(FtBaseModel):
    diagnostic_type: str
    node_id: str
    passed: bool
    details: str
