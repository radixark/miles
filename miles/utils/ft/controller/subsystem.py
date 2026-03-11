from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Annotated, Literal, Union

from pydantic import Field

from miles.utils.ft.adapters.types import SubsystemActuatorProtocol
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.utils.base_model import FtBaseModel
from miles.utils.ft.utils.state_machine import StateMachine


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


@dataclass
class SubsystemEntry:
    """Complete definition of a single subsystem.

    Each subsystem has its own state machine, actuator, detectors, and monitoring config.
    In training-only mode, the dict contains a single "training" entry.
    """

    name: str
    state_machine: StateMachine  # Generic params (SubsystemState, SubsystemContext) defined in M5
    actuator: SubsystemActuatorProtocol
    detectors: list[BaseFaultDetector] = field(default_factory=list)
    monitoring_config: MonitoringIterationProgressConfig | MonitoringSustainedAliveConfig = field(
        default_factory=MonitoringIterationProgressConfig
    )
    get_active_node_ids: Callable[[], set[str]] = field(default_factory=lambda: lambda: set())
