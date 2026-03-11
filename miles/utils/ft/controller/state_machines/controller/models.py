from __future__ import annotations

from pydantic import ConfigDict

from miles.utils.ft.controller.subsystem import SubsystemEntry
from miles.utils.ft.utils.base_model import FtBaseModel


class ControllerState(FtBaseModel):
    model_config = ConfigDict(frozen=True)


class NormalState(ControllerState):
    """All subsystems running normally; step each sub-SM every tick."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    subsystems: dict[str, SubsystemEntry]


class RestartingMainJobState(ControllerState):
    """Waiting for the main job restart to complete."""

    requestor_name: str
