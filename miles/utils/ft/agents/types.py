"""Agent-layer pure data types (no Protocols)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from miles.utils.ft.utils.base_model import FtBaseModel
from miles.utils.ft.utils.diagnostic_types import (
    DiagnosticPipelineResult,
    DiagnosticResult,
    UnknownDiagnosticError,
)

__all__ = [
    "CounterSample",
    "DiagnosticPipelineResult",
    "DiagnosticResult",
    "GaugeSample",
    "MetricSample",
    "UnknownDiagnosticError",
]


class _MetricSampleBase(FtBaseModel):
    name: str
    labels: dict[str, str]


class GaugeSample(_MetricSampleBase):
    value: float
    metric_type: Literal["gauge"] = "gauge"


class CounterSample(_MetricSampleBase):
    delta: float
    metric_type: Literal["counter"] = "counter"


MetricSample = Annotated[
    GaugeSample | CounterSample,
    Field(discriminator="metric_type"),
]
