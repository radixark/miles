from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.runtime_config import ControllerRuntimeConfig
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle

if TYPE_CHECKING:
    from typing import Any

    from miles.utils.ft.adapters.types import NotifierProtocol
    from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol


@dataclass
class TestbedNodeConfig:
    node_id: str
    num_ranks: int = 8
    diagnostic_pass: bool = True


@dataclass
class TestbedConfig:
    training_nodes: list[TestbedNodeConfig]
    rollout_nodes: list[TestbedNodeConfig] = field(default_factory=list)
    spare_nodes: list[str] = field(default_factory=list)
    rollout_num_cells: int = 0

    tick_interval: float = 0.1
    scrape_interval_seconds: float = 0.5
    step_interval: float = 0.1
    health_check_interval: float = 0.5

    detectors: list[BaseFaultDetector] | None = None
    recovery_cooldown: SlidingWindowThrottle | None = None
    monitoring_success_iterations: int | None = None
    monitoring_timeout_seconds: int | None = None
    rollout_alive_threshold_seconds: float | None = None
    rollout_monitoring_alive_duration_seconds: float | None = None

    registration_grace_ticks: int | None = None
    max_simultaneous_bad_nodes: int | None = None
    recovery_timeout_seconds: int | None = None
    notifier_override: NotifierProtocol | None = None
    diagnostic_orchestrator_override: DiagnosticOrchestratorProtocol | None = None
    initial_stable_iterations: int = 2

    def build_runtime_config(self) -> ControllerRuntimeConfig:
        kwargs: dict[str, Any] = {"tick_interval": self.tick_interval}
        if self.monitoring_success_iterations is not None:
            kwargs["monitoring_success_iterations"] = self.monitoring_success_iterations
        if self.monitoring_timeout_seconds is not None:
            kwargs["monitoring_timeout_seconds"] = self.monitoring_timeout_seconds
        if self.registration_grace_ticks is not None:
            kwargs["registration_grace_ticks"] = self.registration_grace_ticks
        if self.max_simultaneous_bad_nodes is not None:
            kwargs["max_simultaneous_bad_nodes"] = self.max_simultaneous_bad_nodes
        if self.recovery_timeout_seconds is not None:
            kwargs["recovery_timeout_seconds"] = self.recovery_timeout_seconds
        if self.recovery_cooldown is not None:
            kwargs["recovery_cooldown_window_minutes"] = self.recovery_cooldown.window_minutes
            kwargs["recovery_cooldown_max_count"] = self.recovery_cooldown.max_count
        if self.rollout_alive_threshold_seconds is not None:
            kwargs["rollout_alive_threshold_seconds"] = self.rollout_alive_threshold_seconds
        if self.rollout_monitoring_alive_duration_seconds is not None:
            kwargs["rollout_monitoring_alive_duration_seconds"] = self.rollout_monitoring_alive_duration_seconds
        return ControllerRuntimeConfig(**kwargs)
