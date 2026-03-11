from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import Field

from miles.utils.ft.adapters.types import SubsystemActuatorProtocol
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.utils.base_model import FtBaseModel


class MonitoringIterationProgressConfig(FtBaseModel):
    """Training mode: confirm recovery after N successful iterations."""

    mode: Literal["iteration_progress"] = "iteration_progress"
    success_iterations: int = 10
    timeout_seconds: int = 600


class MonitoringSustainedAliveConfig(FtBaseModel):
    """Rollout mode: confirm recovery after get_status() == RUNNING for N seconds."""

    mode: Literal["sustained_alive"] = "sustained_alive"
    alive_duration_seconds: int = 180
    timeout_seconds: int = 600


MonitoringConfig = Annotated[
    Union[MonitoringIterationProgressConfig, MonitoringSustainedAliveConfig],
    Field(discriminator="mode"),
]


class RestartMode(Enum):
    SUBSYSTEM = "subsystem"
    MAIN_JOB = "main_job"


@dataclass
class SubsystemConfig:
    """Static configuration for a single subsystem (no runtime state)."""

    actuator: SubsystemActuatorProtocol
    restart_mode: RestartMode = RestartMode.SUBSYSTEM
    detectors: list[BaseFaultDetector] = field(default_factory=list)
    monitoring_config: MonitoringIterationProgressConfig | MonitoringSustainedAliveConfig = field(
        default_factory=MonitoringIterationProgressConfig
    )
    get_active_node_ids: Callable[[], set[str]] = field(default_factory=lambda: lambda: set())
