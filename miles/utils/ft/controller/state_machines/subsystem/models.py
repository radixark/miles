from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from datetime import datetime

from pydantic import ConfigDict, Field

from miles.utils.ft.adapters.types import JobStatus, NotifierProtocol
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.state_machines.recovery.models import RecoveryContext, RecoveryState
from miles.utils.ft.controller.state_machines.restart.models import MonitoringIterationProgressConfig, MonitoringSustainedAliveConfig
from miles.utils.ft.controller.types import MetricStore, TriggerType
from miles.utils.ft.utils.base_model import FtBaseModel
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter, SlidingWindowThrottle


class SubsystemState(FtBaseModel):
    model_config = ConfigDict(frozen=True)


class DetectingAnomalySt(SubsystemState):
    pass


class RecoveringSt(SubsystemState):
    recovery: RecoveryState
    trigger: TriggerType
    recovery_start_time: datetime
    known_bad_node_ids: tuple[str, ...] = ()


class NotifyDeduplicator:
    """Tracks active NOTIFY_HUMAN deduplicator IDs within a subsystem.

    Same problem persisting across ticks: only notify on first occurrence.
    Problem disappears then reappears: notify again.
    Different problems: notify each independently.
    """

    def __init__(self) -> None:
        self._active_ids: set[str] = set()

    def should_notify(self, dedup_id: str | None) -> bool:
        if dedup_id is None:
            return True
        return dedup_id not in self._active_ids

    def sync_active_ids(self, current_ids: set[str]) -> None:
        self._active_ids = current_ids

    @property
    def active_ids(self) -> frozenset[str]:
        return frozenset(self._active_ids)


class SubsystemContext(FtBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # per-tick
    job_status: JobStatus
    tick_count: int
    should_run_detectors: bool
    detector_context: DetectorContext | None

    # deps
    notifier: NotifierProtocol | None
    detectors: list[BaseFaultDetector]
    cooldown: SlidingWindowThrottle
    detector_crash_tracker: SlidingWindowCounter
    recovery_stepper: Callable[..., AsyncGenerator[RecoveryState, None]]
    recovery_context_factory: Callable[[TriggerType, datetime], RecoveryContext]
    on_recovery_duration: Callable[[float], None] | None
    max_simultaneous_bad_nodes: int
    monitoring_config: MonitoringIterationProgressConfig | MonitoringSustainedAliveConfig
    metric_store: MetricStore
    notify_deduplicator: NotifyDeduplicator = Field(default_factory=NotifyDeduplicator)
