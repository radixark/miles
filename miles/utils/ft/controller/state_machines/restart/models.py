from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import ConfigDict, Field, model_validator

from miles.utils.ft.adapters.types import MainJobProtocol, NodeManagerProtocol, NotifierProtocol, SubsystemActuatorProtocol
from miles.utils.ft.controller.types import MetricStore
from miles.utils.ft.utils.base_model import FtBaseModel


class MonitoringIterationProgressConfig(FtBaseModel):
    """Training mode: confirm recovery after N successful iterations."""

    mode: Literal["iteration_progress"] = "iteration_progress"
    success_iterations: int = Field(default=10, ge=0)
    timeout_seconds: int = Field(default=600, ge=0)


class MonitoringSustainedAliveConfig(FtBaseModel):
    """Rollout mode: confirm recovery after get_status() == RUNNING for N seconds."""

    mode: Literal["sustained_alive"] = "sustained_alive"
    alive_duration_seconds: int = Field(default=180, ge=0)
    timeout_seconds: int = Field(default=600, ge=0)


MonitoringConfig = Annotated[
    Union[MonitoringIterationProgressConfig, MonitoringSustainedAliveConfig],
    Field(discriminator="mode"),
]


class ExternalExecutionResult(Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"


class RestartState(FtBaseModel):
    model_config = ConfigDict(frozen=True)
    bad_node_ids: tuple[str, ...] = ()


class EvictingSt(RestartState):
    pass


class StoppingAndRestartingSt(RestartState):
    submitted: bool = False
    submit_time: datetime | None = None

    @model_validator(mode="after")
    def _submit_time_required_when_submitted(self) -> StoppingAndRestartingSt:
        if self.submitted and self.submit_time is None:
            raise ValueError("submit_time must be set when submitted=True")
        return self


class MonitoringProgressSt(RestartState):
    start_time: datetime
    base_iteration: int


class RestartDoneSt(RestartState):
    pass


class RestartFailedSt(RestartState):
    pass


class ExternalRestartingMainJobSt(RestartState):
    """restart_mode is MAIN_JOB; wait for external main job restart."""

    external_execution_result: ExternalExecutionResult | None = None


class RestartContext(FtBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    node_manager: NodeManagerProtocol
    main_job: MainJobProtocol
    metric_store: MetricStore
    notifier: NotifierProtocol | None
    on_new_run: Callable[[str], None] | None
    node_metadata: dict[str, dict[str, str]] = Field(default_factory=dict)

    actuator: SubsystemActuatorProtocol
    monitoring_config: MonitoringIterationProgressConfig | MonitoringSustainedAliveConfig
    is_main_job_restart: bool
    pending_timeout_seconds: int = Field(default=300, ge=0)
    restart_lock: asyncio.Lock | None = None
    on_node_evicted: Callable[[str], None] | None = None
