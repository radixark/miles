"""Controller-layer data types, internal interfaces, and metric name constants."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import polars as pl
from pydantic import Field, model_validator

from miles.utils.ft.agents.types import DiagnosticPipelineResult
from miles.utils.ft.utils.base_model import FtBaseModel

if TYPE_CHECKING:
    from miles.utils.ft.adapters.types import ClusterExecutorProtocol


# ---------------------------------------------------------------------------
# Controller status
# ---------------------------------------------------------------------------

class ControllerMode(str, Enum):
    MONITORING = "monitoring"
    RECOVERY = "recovery"


class ControllerStatus(FtBaseModel):
    mode: ControllerMode
    recovery_phase: str | None
    phase_history: list[str] | None
    tick_count: int
    active_run_id: str | None
    bad_nodes: list[str]
    recovery_in_progress: bool
    bad_nodes_confirmed: bool
    latest_iteration: int | None


# ---------------------------------------------------------------------------
# Fault / decision types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    NONE = "none"
    ENTER_RECOVERY = "enter_recovery"
    NOTIFY_HUMAN = "notify_human"


class TriggerType(str, Enum):
    HANG = "hang"
    NAN_LOSS = "nan_loss"
    CRASH = "crash"
    HARDWARE = "hardware"
    NETWORK = "network"
    MISC = "misc"


class NodeFault(FtBaseModel):
    node_id: str
    reason: str
    ephemeral: bool = False


class Decision(FtBaseModel):
    action: ActionType
    bad_node_ids: list[str] = Field(default_factory=list)
    reason: str
    trigger: TriggerType | None = None

    @model_validator(mode="after")
    def _validate_trigger(self) -> Decision:
        if self.action != ActionType.NONE and self.trigger is None:
            raise ValueError(f"trigger is required when action={self.action.value}")
        return self

    @classmethod
    def no_fault(cls, reason: str) -> Decision:
        return cls(action=ActionType.NONE, reason=reason)

    @classmethod
    def from_node_faults(
        cls,
        faults: list[NodeFault],
        *,
        fallback_reason: str,
        trigger: TriggerType,
    ) -> Decision:
        if not faults:
            return cls(action=ActionType.NONE, reason=fallback_reason)

        non_ephemeral = [f for f in faults if not f.ephemeral]
        if not non_ephemeral:
            return cls(action=ActionType.NONE, reason=f"ephemeral only: {fallback_reason}")

        return cls(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=sorted(unique_node_ids(non_ephemeral)),
            reason="; ".join(f.reason for f in faults),
            trigger=trigger,
        )


def unique_node_ids(faults: list[NodeFault]) -> list[str]:
    """Return deduplicated node IDs from faults, preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for fault in faults:
        if fault.node_id not in seen:
            seen.add(fault.node_id)
            result.append(fault.node_id)
    return result


def filter_node_ids_by_active(node_ids: list[str], active_node_ids: set[str]) -> list[str]:
    """Keep only node IDs that are in the active training placement."""
    return [n for n in node_ids if n in active_node_ids]


# ---------------------------------------------------------------------------
# Training metric types
# ---------------------------------------------------------------------------

class StepValue(NamedTuple):
    step: int
    value: float


class TimedStepValue(NamedTuple):
    step: int
    timestamp: datetime
    value: float


# ---------------------------------------------------------------------------
# Controller-internal protocols (metric store, diagnostics)
# ---------------------------------------------------------------------------

@runtime_checkable
class DiagnosticOrchestratorProtocol(Protocol):
    async def run_diagnostic_pipeline(
        self,
        pre_executors: list[ClusterExecutorProtocol] | None = None,
    ) -> DiagnosticPipelineResult: ...


@runtime_checkable
class MetricQueryProtocol(Protocol):
    def query_latest(
        self,
        metric_name: str,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def query_range(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def changes(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def count_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def avg_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...


@runtime_checkable
class MetricStoreLifecycle(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


@runtime_checkable
class MetricStoreProtocol(MetricQueryProtocol, MetricStoreLifecycle, Protocol): ...


@runtime_checkable
class ScrapeTargetManagerProtocol(Protocol):
    def add_scrape_target(self, target_id: str, address: str) -> None: ...

    def remove_scrape_target(self, target_id: str) -> None: ...


@runtime_checkable
class TrainingMetricStoreProtocol(Protocol):
    def latest(self, metric_name: str) -> float | None: ...

    def query_last_n_steps(
        self,
        metric_name: str,
        last_n: int,
    ) -> list[StepValue]: ...

    def query_time_window(
        self,
        metric_name: str,
        window: timedelta,
    ) -> list[TimedStepValue]: ...
