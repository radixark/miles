from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import ConfigDict, Field

from miles.utils.ft.adapters.types import MainJobProtocol, NodeManagerProtocol, NotifierProtocol, SubsystemActuatorProtocol
from miles.utils.ft.controller.subsystem_hub import RestartMode
from miles.utils.ft.controller.types import MetricStore
from miles.utils.ft.utils.base_model import FtBaseModel


class MonitoringIterationProgressConfig(FtBaseModel):
    """Training mode: confirm recovery after N successful iterations."""

    mode: Literal["iteration_progress"] = "iteration_progress"
    success_iterations: int = 10
    timeout_seconds: int = 600


class MonitoringSustainedAliveConfig(FtBaseModel):
    """Rollout mode: confirm recovery after get_status() == RUNNING for N seconds."""

    mode: Literal["sustained_alive"] = "sustained_alive"
    alive_duration_seconds: int = 180
    timeout_seconds: int = 600


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
    on_main_job_new_run: Callable[[str], None] | None
    node_metadata: dict[str, dict[str, str]] = Field(default_factory=dict)

    actuator: SubsystemActuatorProtocol
    monitoring_config: MonitoringIterationProgressConfig | MonitoringSustainedAliveConfig
    restart_mode: RestartMode
