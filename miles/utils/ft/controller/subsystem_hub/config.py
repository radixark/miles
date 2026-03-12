from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from miles.utils.ft.adapters.types import SubsystemActuatorProtocol
from miles.utils.ft.controller.detectors.base import BaseFaultDetector

if TYPE_CHECKING:
    from miles.utils.ft.controller.state_machines.restart.models import (
        MonitoringIterationProgressConfig,
        MonitoringSustainedAliveConfig,
    )


class RestartMode(Enum):
    SUBSYSTEM = "subsystem"
    MAIN_JOB = "main_job"


def _default_monitoring_config() -> MonitoringIterationProgressConfig:
    from miles.utils.ft.controller.state_machines.restart.models import MonitoringIterationProgressConfig

    return MonitoringIterationProgressConfig()


@dataclass
class SubsystemConfig:
    """Static configuration for a single subsystem (no runtime state)."""

    actuator: SubsystemActuatorProtocol
    restart_mode: RestartMode = RestartMode.SUBSYSTEM
    detectors: list[BaseFaultDetector] = field(default_factory=list)
    monitoring_config: MonitoringIterationProgressConfig | MonitoringSustainedAliveConfig = field(
        default_factory=_default_monitoring_config
    )
    get_active_node_ids: Callable[[], set[str]] = field(default_factory=lambda: lambda: set())
