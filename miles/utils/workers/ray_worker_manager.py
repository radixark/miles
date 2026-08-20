import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import ray

from miles.utils.http_utils import _wrap_ipv6
from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_spec import (
    BaseWorkerSpec,
    CommandWorkerSpec,
    HostAndPort,
    LaunchCommandContext,
    NamedHostAndPorts,
)

# TODO: unique name, maybe with args.run_uuid
_ACTOR_NAME = "ray_worker_manager"


class RayWorkerManager:
    def __init__(self):
        self.port_allocator = PortAllocator()

    @staticmethod
    def launch(specs: list[BaseWorkerSpec], pgs: dict[str, Any]):
        obj = ray.remote(RayWorkerManager).options(name=_ACTOR_NAME).remote()
        ray.get(obj.init.remote(specs, pgs))
        return obj

    @staticmethod
    def get_handle() -> ray.actor.ActorHandle:
        return ray.get_actor(_ACTOR_NAME)

    async def init(self, specs: list[BaseWorkerSpec], pgs: dict[str, Any]):
        self._pools = {spec.name: _PoolManager.initial(spec, self) for spec in specs}
        assert len(self._pools) == len(specs)

        await self._for_all_worker_managers(lambda a: a.launch_actor())
        await self._for_all_worker_managers(lambda a: a.alloc_ports())
        await self._for_all_worker_managers(lambda a: a.post_setup())

    def get_worker_addr(self, worker_name: str) -> HostAndPort:
        matches = [
            a.primary_addr
            for g in self._pools.values()
            for c in g.cells
            for a in c.actors
            if a.name == worker_name
        ]
        assert len(matches) == 1, f"{matches=}"
        return matches[0]

    async def _for_all_worker_managers(self, fn: Callable[["_BaseActorManager"], Any]):
        await asyncio.gather(*[fn(a) for g in self._pools.values() for c in g.cells for a in c.actors])


@dataclass(kw_only=True)
class _PoolManager:
    spec: BaseWorkerSpec
    cells: list["_CellManager"]

    @classmethod
    def initial(cls, spec: BaseWorkerSpec, manager_ref: RayWorkerManager) -> "_PoolManager":
        return cls(
            spec=spec,
            cells=[
                _CellManager(
                    cell_index=cell_index,
                    actors=[
                        # TODO support Serve mode
                        _CommandActorManager(
                            cell_index=cell_index,
                            worker_in_cell_index=worker_in_cell_index,
                            manager_ref=manager_ref,
                            spec=spec,
                            actor_handle=None,
                            generation=1,
                        )
                        for worker_in_cell_index in range(spec.scheduling.num_workers_per_cell)
                    ],
                )
                for cell_index in range(spec.scheduling.num_cells)
            ],
        )


@dataclass(kw_only=True)
class _CellManager:
    cell_index: int
    actors: list["_BaseActorManager"]


SpecT = TypeVar("SpecT", bound=BaseWorkerSpec)


@dataclass(kw_only=True)
class _BaseActorManager(Generic[SpecT]):
    manager_ref: RayWorkerManager
    cell_index: int
    worker_in_cell_index: int
    spec: SpecT
    actor_handle: ray.actor.ActorHandle | None
    self_addrs: NamedHostAndPorts | None = None
    generation: int

    async def launch_actor(self) -> None:
        raise NotImplementedError

    async def alloc_ports(self) -> None:
        raise NotImplementedError

    async def post_setup(self) -> None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return compute_worker_name(
            pool_id=self.spec.name,
            cell_index=self.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
        )

    @property
    def primary_addr(self) -> HostAndPort:
        return self.self_addrs["primary"]


@dataclass
class _CommandActorManager(_BaseActorManager[CommandWorkerSpec]):
    async def launch_actor(self) -> None:
        self.actor_handle = (
            ray.remote(CommandActor)
            .options(
                # TODO generalize
                num_cpus=0.2,
                num_gpus=0,
                runtime_env={"env_vars": self.spec.env_var()},
            )
            .remote()
        )

    async def alloc_ports(self) -> None:
        self.self_addrs = {}

        node_ip = await self.actor_handle._get_node_ip.remote()
        for port_info in self.spec.port_infos:
            if port_info.allow_dynamic:
                port = self.manager_ref.port_allocator.alloc(
                    self.actor_handle, node_ip=node_ip, consecutive=port_info.num_consecutive
                )
            else:
                port = port_info.static_port + (self.cell_index if port_info.offset_by_cell else 0)
                await self._assert_static_port_is_free(port=port, port_name=port_info.name, node_ip=node_ip)
            self.self_addrs[port_info.name] = HostAndPort(host=_wrap_ipv6(node_ip), port=port)

    async def _assert_static_port_is_free(self, *, port: int, port_name: str, node_ip: str) -> None:
        free = await self.actor_handle._is_port_available.remote(port=port)
        assert free, (
            f"Port {port} on {node_ip} is already in use, so {self.name} cannot serve its {port_name!r} "
            f"endpoint there; a stale process from an earlier run is the usual cause"
        )

    async def post_setup(self) -> None:
        ctx = LaunchCommandContext(
            cell_index=self.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
            self_addrs=self.self_addrs,
            pool_addrs={},  # TODO
            gpu_ids=[],  # TODO
        )
        launch_cmd = self.spec.launch_command(ctx)
        self.actor_handle.run.remote(cmd=launch_cmd, envs={})
