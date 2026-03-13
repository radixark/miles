from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from miles.utils.ft.adapters.types import SubsystemActuatorProtocol
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle

if TYPE_CHECKING:
    from miles.utils.ft.controller.state_machines.restart.models import (
        MonitoringIterationProgressConfig,
        MonitoringRunningAfterDelayConfig,
    )


class RestartMode(Enum):
    SUBSYSTEM = "subsystem"
    MAIN_JOB = "main_job"


def _default_monitoring_config() -> MonitoringIterationProgressConfig:
    from miles.utils.ft.controller.state_machines.restart.models import MonitoringIterationProgressConfig

    return MonitoringIterationProgressConfig()


@dataclass
class SubsystemConfig:
    """Pure static configuration — no runtime state or closures."""

    restart_mode: RestartMode = RestartMode.SUBSYSTEM
    detectors: list[BaseFaultDetector] = field(default_factory=list)
    monitoring_config: MonitoringIterationProgressConfig | MonitoringRunningAfterDelayConfig = field(
        default_factory=_default_monitoring_config
    )


@dataclass
class SubsystemRuntime:
    """Runtime dependencies — stateful objects and closures."""

    actuator: SubsystemActuatorProtocol
    cooldown: SlidingWindowThrottle = field(default_factory=lambda: SlidingWindowThrottle(window_minutes=30.0, max_count=3))
    get_active_node_ids: Callable[[], frozenset[str]] = field(default_factory=lambda: lambda: frozenset())


@dataclass
class SubsystemSpec:
    """Complete specification for a single subsystem: config + runtime."""

    config: SubsystemConfig
    runtime: SubsystemRuntime
