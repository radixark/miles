from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from datetime import datetime

from pydantic import ConfigDict

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
    known_bad_node_ids: list[str] = []


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
