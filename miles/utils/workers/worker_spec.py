from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Self

from pydantic import ConfigDict, model_validator

from miles.utils.args.runtime_base import BaseLeafConfig
from miles.utils.math_utils import exact_div
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.types import DeployComponent, PlatformAccess

RPC_PORT_NAME = "rpc"
MASTER_PORT_NAME = "master"
DEFAULT_RPC_PORT = 8000


class PortInfo(FrozenStrictBaseModel):
    name: str
    static_port: int
    mode: Literal["per_worker", "master"] = "per_worker"
    allow_dynamic: bool = False
    num_consecutive: int = 1
    offset_by_cell: bool = False

    def effective_static_port(self, *, worker_in_pod_index: int) -> int:
        if self.mode == "per_worker":
            return self.static_port + worker_in_pod_index * self.num_consecutive
        return self.static_port

    @model_validator(mode="after")
    def _reject_offsetting_a_dynamically_allocated_port(self) -> "PortInfo":
        assert not (
            self.offset_by_cell and self.allow_dynamic
        ), f"Port {self.name!r} cannot be offset by cell index: it is allocated dynamically"
        return self


DEFAULT_RPC_PORT_INFO = PortInfo(
    name=RPC_PORT_NAME,
    static_port=DEFAULT_RPC_PORT,
    mode="per_worker",
    allow_dynamic=True,
)


class SchedulingSpec(FrozenStrictBaseModel):
    num_cells: int
    num_workers_per_cell: int
    num_gpus_per_worker: float
    num_cpus_per_worker: float = 0.2
    num_gpu_slots_per_worker: int = 0
    num_gpus_per_node: int = 0
    pg_name: str | None = None
    pg_slot_offset: int = 0
    pin_to_head: bool = False

    def gpus_per_cell(self) -> int:
        return self.num_workers_per_cell * self.num_gpu_slots_per_worker

    def pods_per_cell(self) -> int:
        gpus_per_cell = self.gpus_per_cell()
        if gpus_per_cell <= self.num_gpus_per_node:
            return 1
        return exact_div(gpus_per_cell, self.num_gpus_per_node)

    def gpus_per_pod(self) -> int:
        return exact_div(self.gpus_per_cell(), self.pods_per_cell())

    def workers_per_pod(self) -> int:
        return exact_div(self.num_workers_per_cell, self.pods_per_cell())

    @classmethod
    def single(cls, num_gpus_per_worker: float, pin_to_head: bool = False) -> "SchedulingSpec":
        return SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=num_gpus_per_worker,
            pin_to_head=pin_to_head,
        )


# TODO: improve meta computation logic later
class WorkerMetaContext(FrozenStrictBaseModel):
    cell_index: int


class WorkerLaunchContext(FrozenStrictBaseModel):
    args: Any
    cell_index: int
    worker_in_cell_index: int
    gpu_ids: list[int]


class WorkerCtorContext(WorkerLaunchContext):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    capability: BackendCapability


class BaseSpec(FrozenStrictBaseModel, ABC):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    args: Any
    name: str
    category: str | None = None
    port_infos: list[PortInfo]
    scheduling: SchedulingSpec
    deploy_component: DeployComponent = DeployComponent.PRIMARY
    platform_access: PlatformAccess = PlatformAccess.NONE

    @classmethod
    @abstractmethod
    def slice_configs(cls, args: Any) -> list[Any]: ...

    @classmethod
    @abstractmethod
    def create(cls, config: Any) -> Self | list[Self]: ...

    def meta(self, ctx: WorkerMetaContext) -> dict[str, Any]:
        return {}

    def env_var(self, ctx: WorkerLaunchContext) -> dict[str, str]:
        return {}


class HostAndPort(FrozenStrictBaseModel):
    host: str
    port: int

    @property
    def addr(self):
        return f"http://{self.host}:{self.port}"


# dict key: name
NamedHostAndPorts = dict[str, HostAndPort]


class LaunchCommandContext(WorkerLaunchContext):
    self_addrs: NamedHostAndPorts
    pool_addrs: dict[str, list[NamedHostAndPorts]]
    local_gpu_ids: list[int]


class BaseCommandSpec(BaseSpec):
    @classmethod
    def slice_configs(cls, args: Any) -> list[Any]:
        return [args]

    @abstractmethod
    def launch_command(self, ctx: LaunchCommandContext) -> str: ...


class BaseServeSpec(BaseSpec):
    worker_type: ClassVar[str]
    config_class: ClassVar[type[BaseLeafConfig]]
    worker_class: str
    port_infos: list[PortInfo] = [DEFAULT_RPC_PORT_INFO]
    concurrency_groups: dict[str, int] | None = None

    @classmethod
    def slice_configs(cls, args: Any) -> list[BaseLeafConfig]:
        return [cls.config_class.slice_from(args)]

    @classmethod
    @abstractmethod
    def create(cls, config: Any) -> Self: ...

    @abstractmethod
    def ctor_kwargs(self, ctx: WorkerCtorContext) -> dict[str, Any]: ...
