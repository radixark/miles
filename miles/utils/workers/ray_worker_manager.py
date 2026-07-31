from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.http_utils import _wrap_ipv6
from miles.utils.ray_utils import compute_ray_pin_head_options
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

if TYPE_CHECKING:
    from miles.ray.placement_group import PlacementGroupInfo

# TODO: unique name, maybe with args.run_uuid
_ACTOR_NAME = "ray_worker_manager"


class RayWorkerManager:
    def __init__(self):
        self.port_allocator = PortAllocator()

    @staticmethod
    def launch(specs: list[BaseWorkerSpec], pgs: dict[str, PlacementGroupInfo]):
        obj = ray.remote(RayWorkerManager).options(name=_ACTOR_NAME).remote()
        ray.get(obj.init.remote(specs, pgs))
        return obj

    @staticmethod
    def get_handle() -> ray.actor.ActorHandle:
        return ray.get_actor(_ACTOR_NAME)

    async def init(self, specs: list[BaseWorkerSpec], pgs: dict[str, PlacementGroupInfo]):
        self.pgs = pgs
        self._pools = {spec.name: _PoolManager.initial(spec, self) for spec in specs}
        assert len(self._pools) == len(specs)

        await self._for_all_cells(lambda a: a.launch_actors())
        await self._for_all_cells(lambda a: a.alloc_ports())
        await self._for_all_cells(lambda a: a.post_setup())

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

    def get_addrs(self) -> dict[str, list[NamedHostAndPorts]]:
        return {name: [a.self_addrs for c in g.cells for a in c.actors] for name, g in self._pools.items()}

    async def _for_all_cells(self, fn: Callable[[_CellManager], Any]):
        await asyncio.gather(*[fn(c) for g in self._pools.values() for c in g.cells])


@dataclass(kw_only=True)
class _PoolManager:
    spec: BaseWorkerSpec
    cells: list[_CellManager]

    @classmethod
    def initial(cls, spec: BaseWorkerSpec, manager: RayWorkerManager) -> _PoolManager:
        return cls(
            spec=spec,
            cells=[
                _CellManager(
                    manager=manager,
                    cell_index=cell_index,
                    spec=spec,
                    actors=None,
                )
                for cell_index in range(spec.scheduling.num_cells)
            ],
        )


SpecT = TypeVar("SpecT", bound=BaseWorkerSpec)


@dataclass(kw_only=True)
class _CellManager(Generic[SpecT]):
    manager: RayWorkerManager
    cell_index: int
    spec: SpecT
    actors: list[_BaseActorManager] | None

    async def launch_actors(self):
        assert self.actors is None
        scheduling = self.spec.scheduling
        self.actors = [
            # TODO support Serve mode
            _CommandActorManager(
                manager=self.manager,
                parent=self,
                worker_in_cell_index=worker_in_cell_index,
                spec=self.spec,
                actor_handle=None,
                generation=1,
                gpu_slot_index=(
                    scheduling.pg_slot_offset
                    + (self.cell_index * scheduling.num_workers_per_cell + worker_in_cell_index)
                    * scheduling.num_gpu_slots_per_worker
                    if scheduling.pg_name is not None
                    else None
                ),
            )
            for worker_in_cell_index in range(scheduling.num_workers_per_cell)
        ]
        await self._for_all_actors(lambda a: a.launch_actor())

    async def alloc_ports(self) -> None:
        await self._for_all_actors(lambda a: a.alloc_ports())

    async def post_setup(self) -> None:
        await self._for_all_actors(lambda a: a.post_setup())

    async def _for_all_actors(self, fn: Callable[[_BaseActorManager], Any]):
        await asyncio.gather(*[fn(a) for a in self.actors])


@dataclass(kw_only=True)
class _BaseActorManager(Generic[SpecT]):
    manager: RayWorkerManager
    parent: _CellManager
    worker_in_cell_index: int
    spec: SpecT
    actor_handle: ray.actor.ActorHandle | None
    self_addrs: NamedHostAndPorts | None = None
    generation: int
    gpu_slot_index: int | None

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
            cell_index=self.parent.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
        )

    @property
    def primary_addr(self) -> HostAndPort:
        return self.self_addrs["primary"]

    @property
    def gpu_ids(self) -> list[int]:
        if (pg_name := self.spec.scheduling.pg_name) is None:
            return []
        pg = self.manager.pgs[pg_name]
        base_gpu_id = int(pg.pg_reordered_gpu_ids[self.gpu_slot_index])
        return list(range(base_gpu_id, base_gpu_id + self.spec.scheduling.num_gpu_slots_per_worker))

    @property
    def master_mode_addrs(self) -> NamedHostAndPorts:
        return {info.name: self.self_addrs[info.name] for info in self.spec.port_infos if info.mode == "master"}


@dataclass
class _CommandActorManager(_BaseActorManager[CommandWorkerSpec]):
    async def launch_actor(self) -> None:
        scheduling_strategy = None
        if (pg_name := self.spec.scheduling.pg_name) is not None:
            pg = self.manager.pgs[pg_name]
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg.pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=pg.pg_reordered_bundle_indices[self.gpu_slot_index],
            )

        self.actor_handle = (
            ray.remote(CommandActor)
            .options(
                # TODO generalize
                num_cpus=0.2,
                num_gpus=self.spec.scheduling.num_gpus_per_worker,
                **(dict(scheduling_strategy=s) if (s := scheduling_strategy) is not None else {}),
                runtime_env={"env_vars": self.spec.env_var()},
                **(compute_ray_pin_head_options() if self.spec.scheduling.pin_to_head else {}),
            )
            .remote()
        )

    async def alloc_ports(self) -> None:
        self.self_addrs = {}

        node_ip = await self.actor_handle._get_node_ip.remote()
        for port_info in self.spec.port_infos:
            if self.worker_in_cell_index != 0 and port_info.mode == "master":
                continue
            port = (
                self.manager.port_allocator.alloc(
                    self.actor_handle, node_ip=node_ip, consecutive=port_info.num_consecutive
                )
                if port_info.allow_dynamic
                else port_info.static_port + (self.cell_index if port_info.offset_by_cell else 0)
            )
            self.self_addrs[port_info.name] = HostAndPort(host=_wrap_ipv6(node_ip), port=port)

    async def post_setup(self) -> None:
        ctx = LaunchCommandContext(
            cell_index=self.parent.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
            self_addrs={
                **self.self_addrs,
                **self.parent.actors[0].master_mode_addrs,
            },
            spec_addrs=self.manager.get_addrs(),
            gpu_ids=self.gpu_ids,
        )
        launch_cmd = self.spec.launch_command(ctx)
        self.actor_handle.run.remote(cmd=launch_cmd, envs={})
