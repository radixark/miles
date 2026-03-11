from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

from pydantic import ConfigDict

from miles.utils.ft.adapters.types import JobStatus, NotifierProtocol
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.recovery.models import RecoveryContext, RecoveryState
from miles.utils.ft.controller.subsystem import MonitoringIterationProgressConfig, MonitoringSustainedAliveConfig
from miles.utils.ft.controller.types import TriggerType
from miles.utils.ft.utils.base_model import FtBaseModel
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter, SlidingWindowThrottle


class SubsystemState(FtBaseModel):
    model_config = ConfigDict(frozen=True)


class DetectingAnomaly(SubsystemState):
    pass


class Recovering(SubsystemState):
    recovery: RecoveryState
    trigger: TriggerType
    recovery_start_time: datetime


class RestartingMainJob(SubsystemState):
    """Signal from sub-SM to main SM: requesting main job restart."""


class RestartedMainJob(SubsystemState):
    """Signal from main SM to sub-SM: main job restart completed."""


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
    recovery_stepper: Callable[..., Awaitable[RecoveryState | None]]
    recovery_context_factory: Callable[[TriggerType, datetime], RecoveryContext]
    on_recovery_duration: Callable[[float], None] | None
    max_simultaneous_bad_nodes: int
    monitoring_config: MonitoringIterationProgressConfig | MonitoringSustainedAliveConfig
    mini_wandb: MiniWandb
