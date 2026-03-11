"""Controller-layer data types, internal interfaces, and metric name constants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING

import polars as pl
from pydantic import Field, computed_field, model_validator

from miles.utils.ft.agents.types import DiagnosticPipelineResult
from miles.utils.ft.utils.base_model import FtBaseModel

if TYPE_CHECKING:
    from miles.utils.ft.adapters.types import ClusterExecutorProtocol
    from miles.utils.ft.controller.metrics.mini_wandb import StepValue, TimedStepValue


# ---------------------------------------------------------------------------
# Controller status
# ---------------------------------------------------------------------------


class ControllerMode(str, Enum):
    MONITORING = "monitoring"
    RECOVERY = "recovery"


class RecoveryInfo(FtBaseModel):
    phase: str
    bad_nodes: list[str]
    bad_nodes_confirmed: bool


class ControllerStatus(FtBaseModel):
    tick_count: int
    active_run_id: str | None
    latest_iteration: int | None
    subsystem_states: dict[str, str]
    recovery: RecoveryInfo | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mode(self) -> ControllerMode:
        return ControllerMode.RECOVERY if self.recovery is not None else ControllerMode.MONITORING

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recovery_in_progress(self) -> bool:
        return self.recovery is not None


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
            bad_node_ids=sorted(cls._unique_node_ids(non_ephemeral)),
            reason="; ".join(f.reason for f in faults),
            trigger=trigger,
        )

    @staticmethod
    def _unique_node_ids(faults: list[NodeFault]) -> list[str]:
        """Return deduplicated node IDs from faults, preserving first-seen order."""
        seen: set[str] = set()
        result: list[str] = []
        for fault in faults:
            if fault.node_id not in seen:
                seen.add(fault.node_id)
                result.append(fault.node_id)
        return result


# ---------------------------------------------------------------------------
# Controller-internal protocols (metric store, diagnostics)
# ---------------------------------------------------------------------------


class DiagnosticOrchestratorProtocol(ABC):
    @abstractmethod
    async def run_diagnostic_pipeline(
        self,
        pre_executors: list[ClusterExecutorProtocol] | None = None,
    ) -> DiagnosticPipelineResult: ...


class MetricQueryProtocol(ABC):
    @abstractmethod
    def query_latest(
        self,
        metric_name: str,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def query_range(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def changes(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def count_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def avg_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...


class MetricStoreLifecycle(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...


class MetricStoreProtocol(MetricQueryProtocol, MetricStoreLifecycle): ...


class ScrapeTargetManagerProtocol(ABC):
    @abstractmethod
    def add_scrape_target(self, target_id: str, address: str) -> None: ...

    @abstractmethod
    def remove_scrape_target(self, target_id: str) -> None: ...


class TrainingMetricStoreProtocol(ABC):
    @abstractmethod
    def latest(self, metric_name: str) -> float | None: ...

    @abstractmethod
    def query_last_n_steps(
        self,
        metric_name: str,
        last_n: int,
    ) -> list[StepValue]: ...

    @abstractmethod
    def query_time_window(
        self,
        metric_name: str,
        window: timedelta,
    ) -> list[TimedStepValue]: ...
