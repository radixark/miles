from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from miles.utils.ft.adapters.types import SubsystemActuatorProtocol
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.utils.base_model import FtBaseModel
from miles.utils.ft.utils.state_machine import StateMachine


class MonitoringConfig(FtBaseModel):
    """Subsystem-specific recovery confirmation configuration.

    iteration_progress: training mode — confirm recovery after N successful iterations.
    sustained_alive: rollout mode — confirm recovery after get_status() == RUNNING for N seconds.
    """

    mode: Literal["iteration_progress", "sustained_alive"]

    # mode == "iteration_progress"
    success_iterations: int = 10
    timeout_seconds: int = 600

    # mode == "sustained_alive"
    alive_duration_seconds: int = 180


@dataclass
class SubsystemEntry:
    """Complete definition of a single subsystem.

    Each subsystem has its own state machine, actuator, detectors, and monitoring config.
    In training-only mode, the dict contains a single "training" entry.
    """

    name: str
    state_machine: StateMachine  # Generic params (MainState, MainContext) defined in M5
    actuator: SubsystemActuatorProtocol
    detectors: list[BaseFaultDetector] = field(default_factory=list)
    monitoring_config: MonitoringConfig = field(
        default_factory=lambda: MonitoringConfig(mode="iteration_progress")
    )
    get_active_node_ids: Callable[[], set[str]] = field(default_factory=lambda: lambda: set())
