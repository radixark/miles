from __future__ import annotations

from enum import Enum
from typing import Literal

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.test_utils.fault_injector import FailureMode


class TriState(str, Enum):
    """K8s condition status: ``"True"``, ``"False"``, or ``"Unknown"``."""

    TRUE = "True"
    FALSE = "False"
    UNKNOWN = "Unknown"


class _OkResponse(StrictBaseModel):
    status: str = "ok"


class CellCondition(StrictBaseModel):
    type: Literal["Allocated", "Healthy"]
    status: TriState
    reason: str | None = None
    message: str | None = None
    lastTransitionTime: str | None = None

    @classmethod
    def allocated(cls, status: TriState) -> CellCondition:
        return cls(type="Allocated", status=status)

    @classmethod
    def healthy(cls, status: TriState, *, reason: str | None = None) -> CellCondition:
        return cls(type="Healthy", status=status, reason=reason)


class CellStatus(StrictBaseModel):
    phase: Literal["Pending", "Running", "Suspended"]
    conditions: list[CellCondition]


class CellSpec(StrictBaseModel):
    suspend: bool = False


class CellMetadata(StrictBaseModel):
    name: str
    labels: dict[str, str]


class Cell(StrictBaseModel):
    apiVersion: Literal["miles.io/v1"] = "miles.io/v1"
    kind: Literal["Cell"] = "Cell"
    metadata: CellMetadata
    spec: CellSpec
    status: CellStatus


class CellList(StrictBaseModel):
    apiVersion: Literal["miles.io/v1"] = "miles.io/v1"
    kind: Literal["CellList"] = "CellList"
    items: list[Cell]


class CellPatchSpec(StrictBaseModel):
    suspend: bool | None = None


class CellPatch(StrictBaseModel):
    spec: CellPatchSpec | None = None


class FaultInjection(StrictBaseModel):
    mode: FailureMode
    sub_index: int = 0


class K8sStatus(StrictBaseModel):
    apiVersion: Literal["v1"] = "v1"
    kind: Literal["Status"] = "Status"
    status: Literal["Failure"] = "Failure"
    message: str
    reason: str
    code: int
