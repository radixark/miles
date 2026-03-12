from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

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
from miles.utils.ft.controller.state_machines.subsystem.models import SubsystemState
from miles.utils.ft.controller.subsystem import SubsystemConfig
from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol, TimeSeriesStoreProtocol
from miles.utils.ft.utils.base_model import FtBaseModel
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter, SlidingWindowThrottle


class MainState(FtBaseModel):
    model_config = ConfigDict(frozen=True)


class NormalSt(MainState):
    """All subsystems running normally; step each sub-SM every tick."""

    subsystems: dict[str, SubsystemState]


class RestartingMainJobSt(MainState):
    """Waiting for the main job restart to complete."""

    requestor_name: str
    start_time: datetime
    requestor_frozen_state: SubsystemState


class MainContext(FtBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Level 2 job management
    main_job: MainJobProtocol
    subsystem_configs: dict[str, SubsystemConfig]

    # Shared per-tick data
    tick_count: int
    job_status: JobStatus

    # Shared deps (for building sub-SM contexts)
    metric_store: TimeSeriesStoreProtocol
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
