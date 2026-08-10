from collections.abc import Callable
from typing import Any, Literal

from pydantic import ConfigDict, model_validator

from miles.utils.math_utils import exact_div
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.backend_capability.base import BackendCapability

RPC_PORT_NAME = "rpc"
MASTER_PORT_NAME = "master"
DEFAULT_RPC_PORT = 8000


def _port_info_name(port_info: "PortInfo | dict") -> str:
    return port_info["name"] if isinstance(port_info, dict) else port_info.name


class PortInfo(FrozenStrictBaseModel):
    name: str
    static_port: int
    mode: Literal["per_worker", "master"] = "per_worker"
    allow_dynamic: bool = False
    num_consecutive: int = 1
    offset_by_cell: bool = False

    @model_validator(mode="after")
    def _reject_offsetting_a_dynamically_allocated_port(self) -> "PortInfo":
        assert not (
            self.offset_by_cell and self.allow_dynamic
        ), f"Port {self.name!r} cannot be offset by cell index: it is allocated dynamically"
        return self


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
    cell_index: int
    worker_in_cell_index: int
    gpu_ids: list[int]


class WorkerCtorContext(WorkerLaunchContext):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    capability: BackendCapability


SpecMetaFn = Callable[[WorkerMetaContext], dict[str, Any]]


class BaseWorkerSpec(FrozenStrictBaseModel):
    name: str
    port_infos: list[PortInfo]
    env_var: Callable[[WorkerLaunchContext], dict[str, str]]
    scheduling: SchedulingSpec
    meta: SpecMetaFn | None = None


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


class CommandWorkerSpec(BaseWorkerSpec):
    launch_command: Callable[[LaunchCommandContext], str]


class ServeWorkerSpec(BaseWorkerSpec):
    worker_class: str
    ctor_kwargs: Callable[[WorkerCtorContext], dict[str, Any]]
    concurrency_groups: dict[str, int] | None = None
    method_concurrency_groups: dict[str, str] | None = None

    @model_validator(mode="after")
    def _require_the_groups_and_their_methods_together(self) -> "ServeWorkerSpec":
        assert (self.concurrency_groups is None) == (self.method_concurrency_groups is None), (
            f"Worker {self.name!r} must declare concurrency_groups and method_concurrency_groups "
            f"together: groups nobody is assigned to are dead weight, and a method assigned to a "
            f"group the actor never declares makes Ray reject the actor"
        )
        assert self.method_concurrency_groups is None or set(self.method_concurrency_groups.values()) <= set(
            self.concurrency_groups
        ), (
            f"Worker {self.name!r} routes methods to undeclared concurrency groups: "
            f"{sorted(set(self.method_concurrency_groups.values()) - set(self.concurrency_groups))}"
        )
        return self

    @model_validator(mode="before")
    @classmethod
    def _inject_rpc_port(cls, values: dict) -> dict:
        if "port_infos" not in values:
            return values

        port_infos = list(values["port_infos"])
        if all(_port_info_name(port_info) != RPC_PORT_NAME for port_info in port_infos):
            port_infos.append(
                PortInfo(name=RPC_PORT_NAME, static_port=DEFAULT_RPC_PORT, mode="per_worker", allow_dynamic=True)
            )
        return {**values, "port_infos": port_infos}
