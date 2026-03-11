from __future__ import annotations

from collections.abc import Callable

from pydantic import ConfigDict

from miles.utils.ft.adapters.types import MainJobProtocol
from miles.utils.ft.controller.subsystem import SubsystemEntry
from miles.utils.ft.utils.base_model import FtBaseModel


class ControllerContext(FtBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    main_job: MainJobProtocol
    create_fresh_subsystems: Callable[[], dict[str, SubsystemEntry]]
