from collections.abc import Callable
from typing import Any, Literal

from pydantic import model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel

RPC_PORT_NAME = "rpc"
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
    pg_name: str | None = None
    pg_slot_offset: int = 0
    pin_to_head: bool = False

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


class BaseWorkerSpec(FrozenStrictBaseModel):
    name: str
    port_infos: list[PortInfo]
    env_var: Callable[[WorkerLaunchContext], dict[str, str]]
    scheduling: SchedulingSpec
    meta: Callable[[WorkerMetaContext], dict[str, Any]] | None = None


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


class CommandWorkerSpec(BaseWorkerSpec):
    launch_command: Callable[[LaunchCommandContext], str]


class ServeWorkerSpec(BaseWorkerSpec):
    worker_class: str
    ctor_kwargs: Callable[[WorkerLaunchContext], dict[str, Any]]
    concurrency_groups: dict[str, int] | None = None

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
