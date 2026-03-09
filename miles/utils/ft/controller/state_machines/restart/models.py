from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pydantic import ConfigDict

from miles.utils.ft.adapters.types import NodeManagerProtocol, NotifierProtocol, TrainingJobProtocol
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
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


class RestartContext(FtBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    node_manager: NodeManagerProtocol
    training_job: TrainingJobProtocol
    mini_wandb: MiniWandb
    notifier: NotifierProtocol | None
    on_new_run: Callable[[str], None] | None
    monitoring_success_iterations: int
    monitoring_timeout_seconds: int
