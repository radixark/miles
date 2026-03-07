from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict

from miles.utils.ft.models.base import FtBaseModel


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
