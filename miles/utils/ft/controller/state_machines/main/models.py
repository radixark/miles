from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

from pydantic import ConfigDict

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.state_machines.recovery.models import RecoveryState
from miles.utils.ft.controller.state_machines.recovery.models import RecoveryContext
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import TriggerType
from miles.utils.ft.protocols.platform import JobStatus, NotifierProtocol
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter, SlidingWindowThrottle


class MainState(FtBaseModel):
    model_config = ConfigDict(frozen=True)


class DetectingAnomaly(MainState):
    pass


class Recovering(MainState):
    recovery: RecoveryState
    trigger: TriggerType
    recovery_start_time: datetime


class MainContext(FtBaseModel):
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
