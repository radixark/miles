from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle

if TYPE_CHECKING:
    from miles.utils.ft.adapters.types import NotifierProtocol


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
