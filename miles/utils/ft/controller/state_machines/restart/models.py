from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pydantic import ConfigDict

from miles.utils.ft.adapters.types import MainJobProtocol, NodeManagerProtocol, NotifierProtocol, SubsystemActuatorProtocol
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.subsystem import MonitoringConfig
from miles.utils.ft.utils.base_model import FtBaseModel


class RestartState(FtBaseModel):
    model_config = ConfigDict(frozen=True)
    bad_node_ids: list[str] = []


class Evicting(RestartState):
    pass


class StoppingAndRestarting(RestartState):
    submitted: bool = False
    submit_time: datetime | None = None


class MonitoringProgress(RestartState):
    start_time: datetime
    base_iteration: int


class RestartDone(RestartState):
    pass


class RestartFailed(RestartState):
    pass


class RestartEscalated(RestartState):
    """Subsystem does not support Level 1 restart; escalate to Level 2 (job restart)."""


class RestartContext(FtBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    node_manager: NodeManagerProtocol
    main_job: MainJobProtocol
    mini_wandb: MiniWandb
    notifier: NotifierProtocol | None
    on_new_run: Callable[[str], None] | None
    monitoring_success_iterations: int
    monitoring_timeout_seconds: int
    node_metadata: dict[str, dict[str, str]] = {}

    actuator: SubsystemActuatorProtocol
    monitoring_config: MonitoringConfig
    has_level1_restart: bool
