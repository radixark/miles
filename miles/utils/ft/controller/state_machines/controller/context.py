from __future__ import annotations

from collections.abc import Callable

from pydantic import ConfigDict

from miles.utils.ft.adapters.types import (
    JobStatus,
    MainJobProtocol,
    NodeAgentProtocol,
    NodeManagerProtocol,
    NotifierProtocol,
)
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.subsystem import SubsystemEntry
from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol, MetricStoreProtocol
from miles.utils.ft.utils.base_model import FtBaseModel
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter, SlidingWindowThrottle


class ControllerContext(FtBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Level 2 job management
    main_job: MainJobProtocol
    create_fresh_subsystems: Callable[[], dict[str, SubsystemEntry]]

    # Shared per-tick data
    tick_count: int
    job_status: JobStatus

    # Shared deps (for building sub-SM contexts)
    metric_store: MetricStoreProtocol
    mini_wandb: MiniWandb
    agents: dict[str, NodeAgentProtocol]
    notifier: NotifierProtocol | None
    node_manager: NodeManagerProtocol
    diagnostic_orchestrator: DiagnosticOrchestratorProtocol
    cooldown: SlidingWindowThrottle
    detector_crash_tracker: SlidingWindowCounter
    recovery_timeout_seconds: int
    max_simultaneous_bad_nodes: int

    # Callbacks
    on_new_run: Callable[[str], None] | None
    rank_pids_provider: Callable[[str], dict[int, int]] | None

    # Optional
    controller_exporter: ControllerExporter | None
    on_recovery_duration: Callable[[float], None] | None

    # Grace period
    registration_grace_ticks: int
