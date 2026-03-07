from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict

from miles.utils.ft.controller.recovery.recovery_stepper.states import RecoveryState
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import TriggerType


class MainState(FtBaseModel):
    model_config = ConfigDict(frozen=True)


class DetectingAnomaly(MainState):
    pass


class Recovering(MainState):
    recovery: RecoveryState
    trigger: TriggerType
    recovery_start_time: datetime
