from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict

from miles.utils.ft.controller.subsystem import SubsystemEntry
from miles.utils.ft.utils.base_model import FtBaseModel


class MainState(FtBaseModel):
    model_config = ConfigDict(frozen=True)


class NormalState(MainState):
    """All subsystems running normally; step each sub-SM every tick."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    subsystems: dict[str, SubsystemEntry]


class RestartingMainJobState(MainState):
    """Waiting for the main job restart to complete."""

    requestor_name: str
    start_time: datetime
