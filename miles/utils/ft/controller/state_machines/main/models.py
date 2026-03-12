from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.state_machines.subsystem.models import SubsystemState
from miles.utils.ft.controller.types import SharedDeps
from miles.utils.ft.utils.base_model import FtBaseModel


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

    shared: SharedDeps

    # Per-tick data
    tick_count: int
    run_start_tick: int
    job_status: JobStatus

    # Node metadata (refreshed each tick from NodeAgentRegistry)
    node_metadata: dict[str, dict[str, str]]
