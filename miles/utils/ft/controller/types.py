"""Controller-layer data types, internal interfaces, and metric name constants."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING

import polars as pl
from pydantic import Field, computed_field, model_validator

from miles.utils.ft.adapters.types import MainJobProtocol, NodeManagerProtocol, NotifierProtocol
from miles.utils.ft.utils.base_model import FtBaseModel
from miles.utils.ft.utils.diagnostic_types import DiagnosticPipelineResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from miles.utils.ft.adapters.types import ClusterExecutorProtocol
    from miles.utils.ft.controller.metrics.exporter import ControllerExporter
    from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb, StepValue, TimedStepValue
    from miles.utils.ft.controller.subsystem_hub.config import SubsystemSpec
    from miles.utils.ft.utils.sliding_window import SlidingWindowCounter


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
    recoveries: dict[str, RecoveryInfo] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mode(self) -> ControllerMode:
        return ControllerMode.RECOVERY if self.recoveries else ControllerMode.MONITORING

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recovery_in_progress(self) -> bool:
        return bool(self.recoveries)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recovery(self) -> RecoveryInfo | None:
        return next(iter(self.recoveries.values()), None)


# ---------------------------------------------------------------------------
# Fault / decision types
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    NONE = "none"
    ENTER_RECOVERY = "enter_recovery"
    NOTIFY_HUMAN = "notify_human"


class TriggerType(str, Enum):
    """Fault trigger types, ordered by priority (highest first).

    The enum member order doubles as the merge priority for
    _merge_detector_decisions(): when multiple ENTER_RECOVERY decisions
    arrive, the trigger with the smallest index wins.
    """

    HARDWARE = "hardware"
    CRASH = "crash"
    NAN_LOSS = "nan_loss"
    HANG = "hang"
    NETWORK = "network"
    TELEMETRY_BLIND = "telemetry_blind"
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
    notify_deduplicator_id: str | None = None

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
            logger.debug("controller: from_node_faults no faults, reason=%s", fallback_reason)
            return cls(action=ActionType.NONE, reason=fallback_reason)

        non_ephemeral = [f for f in faults if not f.ephemeral]
        if not non_ephemeral:
            logger.debug(
                "controller: from_node_faults all %d faults ephemeral, reason=%s",
                len(faults),
                fallback_reason,
            )
            return cls(action=ActionType.NONE, reason=f"ephemeral only: {fallback_reason}")

        ephemeral = [f for f in faults if f.ephemeral]
        reason_parts = [f.reason for f in non_ephemeral]
        if ephemeral:
            reason_parts.append(f"(also {len(ephemeral)} ephemeral fault(s) ignored)")

        return cls(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=sorted(cls._unique_node_ids(non_ephemeral)),
            reason="; ".join(reason_parts),
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


class TimeSeriesQueryProtocol(ABC):
    @abstractmethod
    def query_single_latest(
        self,
        metric_name: str,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

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


class TimeSeriesStoreLifecycle(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...


class TimeSeriesStoreProtocol(TimeSeriesQueryProtocol, TimeSeriesStoreLifecycle): ...


class ScrapeTargetManagerProtocol(ABC):
    @abstractmethod
    def add_scrape_target(self, target_id: str, address: str) -> None: ...

    @abstractmethod
    def remove_scrape_target(self, target_id: str) -> None: ...


class NullScrapeTargetManager(ScrapeTargetManagerProtocol):
    def add_scrape_target(self, target_id: str, address: str) -> None:
        pass

    def remove_scrape_target(self, target_id: str) -> None:
        pass


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


@dataclass
class MetricStore:
    time_series_store: TimeSeriesStoreProtocol
    mini_wandb: MiniWandb


class SharedDeps:
    """Stable dependencies that do not change tick-to-tick.

    MainContext used to carry ~18 flat fields mixing stable deps with
    per-tick data. SharedDeps groups the stable portion so MainContext
    only holds per-tick fields plus this container.

    Not a @dataclass to avoid Pydantic resolving its field annotations
    (which would cause circular imports).
    """

    __slots__ = (
        "main_job",
        "subsystem_specs",
        "metric_store",
        "notifier",
        "node_manager",
        "diagnostic_orchestrator",
        "detector_crash_tracker",
        "recovery_timeout_seconds",
        "max_simultaneous_bad_nodes",
        "on_main_job_new_run",
        "rank_pids_provider",
        "controller_exporter",
        "on_recovery_duration",
        "registration_grace_ticks",
        "on_convergence_failure",
        "restart_lock",
    )

    def __init__(
        self,
        *,
        main_job: MainJobProtocol,
        subsystem_specs: dict[str, SubsystemSpec],
        metric_store: MetricStore,
        notifier: NotifierProtocol | None,
        node_manager: NodeManagerProtocol,
        diagnostic_orchestrator: DiagnosticOrchestratorProtocol,
        detector_crash_tracker: SlidingWindowCounter,
        recovery_timeout_seconds: int,
        max_simultaneous_bad_nodes: int,
        on_main_job_new_run: Callable[[str], None] | None,
        rank_pids_provider: Callable[[str], dict[int, int]] | None,
        controller_exporter: ControllerExporter | None,
        on_recovery_duration: Callable[[float], None] | None,
        registration_grace_ticks: int,
        on_convergence_failure: Callable[[object, int], None] | None = None,
        restart_lock: asyncio.Lock | None = None,
    ) -> None:
        self.main_job = main_job
        self.subsystem_specs = subsystem_specs
        self.metric_store = metric_store
        self.notifier = notifier
        self.node_manager = node_manager
        self.diagnostic_orchestrator = diagnostic_orchestrator
        self.detector_crash_tracker = detector_crash_tracker
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.max_simultaneous_bad_nodes = max_simultaneous_bad_nodes
        self.on_main_job_new_run = on_main_job_new_run
        self.rank_pids_provider = rank_pids_provider
        self.controller_exporter = controller_exporter
        self.on_recovery_duration = on_recovery_duration
        self.registration_grace_ticks = registration_grace_ticks
        self.on_convergence_failure = on_convergence_failure
        self.restart_lock = restart_lock
