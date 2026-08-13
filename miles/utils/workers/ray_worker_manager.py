from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.function_registry import load_function
from miles.utils.http_utils import _wrap_ipv6
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import NodeProbeMixin
from miles.utils.ray_utils import compute_ray_pin_head_options
from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.backend_capability.base import BackendCapability, DeferredBackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.rpc.common.metadata import declared_concurrency_groups
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import (
    BaseWorkerSpec,
    CommandWorkerSpec,
    HostAndPort,
    LaunchCommandContext,
    NamedHostAndPorts,
    ServeWorkerSpec,
    WorkerCtorContext,
    WorkerLaunchContext,
    WorkerMetaContext,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from miles.ray.placement_group import PlacementGroupInfo

# TODO: unique name, maybe with args.run_uuid
_ACTOR_NAME = "ray_worker_manager"


class RayWorkerManager:
    def __init__(self):
        self.port_allocator = PortAllocator()

    @staticmethod
    def launch(args, specs: list[BaseWorkerSpec], pgs: dict[str, PlacementGroupInfo]):
        obj = ray.remote(RayWorkerManager).options(name=_ACTOR_NAME).remote()
        ray.get(obj.init.remote(args, specs, pgs))
        return obj

    @staticmethod
    def get_handle() -> ray.actor.ActorHandle:
        return ray.get_actor(_ACTOR_NAME)

    async def init(self, args, specs: list[BaseWorkerSpec], pgs: dict[str, PlacementGroupInfo]):
        configure_logger(args, source=SimpleProcessIdentity(component="worker_manager"))

        self.pgs = pgs
        self._pools = {spec.name: _PoolManager.initial(spec, self) for spec in specs}
        assert len(self._pools) == len(specs)

        await self.start_cells([c.cell_id for c in self._all_cells()])

    async def start_cells(self, cell_ids: list[str]) -> None:
        cells = [cell for cell_id in cell_ids if (cell := self._find_cell(cell_id)).actors is None]
        try:
            await _gather_or_raise([c.launch_actors() for c in cells])
            await _gather_or_raise([c.alloc_ports() for c in cells])
            await _gather_or_raise([c.post_setup() for c in cells])
        except Exception:
            logger.error(f"Starting cells {[c.cell_id for c in cells]} failed, rolling back", exc_info=True)
            await asyncio.gather(*[c.stop() for c in cells], return_exceptions=True)
            raise

    async def stop_cells(self, cell_ids: list[str]) -> None:
        await asyncio.gather(*[self._find_cell(cell_id).stop() for cell_id in cell_ids])

    def inject_fault(self, cell_id: str, *, mode: str, worker_in_cell_index: int) -> None:
        cell = self._find_cell(cell_id)
        if not cell.alive:
            raise RuntimeError(f"Cell {cell_id} is not alive, cannot inject fault")
        if not 0 <= worker_in_cell_index < len(cell.actors):
            raise IndexError(
                f"worker_in_cell_index {worker_in_cell_index} out of range for cell {cell_id} "
                f"(has {len(cell.actors)} workers)"
            )
        cell.actors[worker_in_cell_index].actor_handle.inject_fault.remote(mode)

    def get_worker_addrs(self, worker_name: str) -> NamedHostAndPorts:
        addrs = self._find_actor(worker_name).self_addrs
        assert addrs is not None, (
            f"{worker_name} has not been given its ports yet; a caller reading them now would take the "
            f"endpoints it cannot find for endpoints the worker does not have"
        )
        return addrs

    def get_addrs(self) -> dict[str, list[NamedHostAndPorts]]:
        return {
            name: [a.self_addrs or {} for c in g.cells if c.alive for a in c.actors] for name, g in self._pools.items()
        }

    def get_worker_infos(self, cell_id: str) -> list[WorkerInfo]:
        cell = self._find_cell(cell_id)
        return [self._compute_worker_info(actor) for actor in (cell.actors if cell.actors is not None else [])]

    def get_cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        # TODO: about `get_worker_infos` (which is only used by dashboard)
        unknown = set(pool_ids) - set(self._pools)
        assert not unknown, f"{unknown=} {sorted(self._pools)=}"
        infos = [c.get_info() for name in pool_ids for c in self._pools[name].cells]
        return {info.cell_id: info for info in infos}

    def get_actor_handle(self, worker_name: str) -> ray.actor.ActorHandle:
        return self._find_actor(worker_name).actor_handle

    def _compute_worker_info(self, actor: _BaseActorManager) -> WorkerInfo:
        return WorkerInfo(
            name=actor.name,
            generation=actor.generation,
            self_addrs=actor.self_addrs or {},
            gpu_ids=actor.gpu_ids,
            worker_class=None,
        )

    def _find_actor(self, worker_name: str) -> _BaseActorManager:
        matches = [a for c in self._all_cells() if c.alive for a in c.actors if a.name == worker_name]
        assert len(matches) == 1, f"{matches=}"
        return matches[0]

    def _find_cell(self, cell_id: str) -> _CellManager:
        matches = [c for c in self._all_cells() if c.cell_id == cell_id]
        assert len(matches) == 1, f"{cell_id=} {matches=}"
        return matches[0]

    def _all_cells(self) -> list[_CellManager]:
        return [c for g in self._pools.values() for c in g.cells]


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
    generation: int = 0

    async def launch_actors(self):
        assert self.actors is None
        self.generation += 1
        scheduling = self.spec.scheduling
        actor_manager_cls = _ServeActorManager if isinstance(self.spec, ServeWorkerSpec) else _CommandActorManager
        self.actors = [
            actor_manager_cls(
                manager=self.manager,
                parent=self,
                worker_in_cell_index=worker_in_cell_index,
                spec=self.spec,
                actor_handle=None,
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

    async def stop(self) -> None:
        if self.actors is None:
            return
        await self._for_all_actors(lambda a: a.stop())
        self.actors = None

    async def _for_all_actors(self, fn: Callable[[_BaseActorManager], Any]):
        await asyncio.gather(*[fn(a) for a in self.actors])

    def get_info(self) -> CellInfo:
        return CellInfo(
            cell_id=self.cell_id,
            pool_id=self.spec.name,
            alive=self.alive and self._all_workers_have_addrs,
            worker_names=[a.name for a in self.actors] if self.actors is not None else [],
            workers_hash=f"pseudo-hash-{self.generation}",
            meta=f(WorkerMetaContext(cell_index=self.cell_index)) if (f := self.spec.meta) is not None else {},
        )

    @property
    def cell_id(self) -> str:
        return compute_cell_id(pool_id=self.spec.name, cell_index=self.cell_index)

    @property
    def alive(self) -> bool:
        return self.actors is not None

    @property
    def _all_workers_have_addrs(self) -> bool:
        return all(a.self_addrs is not None for a in self.actors or [])


_SHUTDOWN_TIMEOUT = 30


@dataclass(kw_only=True)
class _BaseActorManager(Generic[SpecT]):
    manager: RayWorkerManager
    parent: _CellManager
    worker_in_cell_index: int
    spec: SpecT
    actor_handle: ray.actor.ActorHandle | None
    self_addrs: NamedHostAndPorts | None = None
    local_gpu_ids: list[int] | None = None
    gpu_slot_index: int | None

    async def launch_actor(self) -> None:
        raise NotImplementedError

    async def post_setup(self) -> None:
        raise NotImplementedError

    async def alloc_ports(self) -> None:
        allocated: NamedHostAndPorts = {}

        node_ip, self.local_gpu_ids = await asyncio.gather(
            self.actor_handle._get_node_ip.remote(),
            self.actor_handle._to_local_gpu_ids.remote(gpu_ids=self.gpu_ids),
        )
        for port_info in self.spec.port_infos:
            if self.worker_in_cell_index != 0 and port_info.mode == "master":
                continue
            if port_info.allow_dynamic:
                port = await self.manager.port_allocator.alloc(
                    self.actor_handle, node_ip=node_ip, consecutive=port_info.num_consecutive
                )
            else:
                port = port_info.static_port + (self.parent.cell_index if port_info.offset_by_cell else 0)
                await self._assert_static_port_is_free(port=port, port_name=port_info.name, node_ip=node_ip)
            allocated[port_info.name] = HostAndPort(host=_wrap_ipv6(node_ip), port=port)

        self.self_addrs = allocated

    async def _assert_static_port_is_free(self, *, port: int, port_name: str, node_ip: str) -> None:
        free = await self.actor_handle._is_port_available.remote(port=port)
        assert free, (
            f"Port {port} on {node_ip} is already in use, so {self.name} cannot serve its {port_name!r} "
            f"endpoint there; a stale process from an earlier run is the usual cause"
        )

    @property
    def launch_context(self) -> WorkerLaunchContext:
        return WorkerLaunchContext(
            cell_index=self.parent.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
            gpu_ids=self.gpu_ids,
        )

    def _compute_remote_options(self) -> dict:
        return {}

    def _create_actor(self, actor_class: type, **ctor_kwargs) -> ray.actor.ActorHandle:
        scheduling_strategy = None
        if (pg_name := self.spec.scheduling.pg_name) is not None:
            pg = self.manager.pgs[pg_name]
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg.pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=pg.pg_reordered_bundle_indices[self.gpu_slot_index],
            )

        remote_options = self._compute_remote_options()
        remote_class = ray.remote(**remote_options)(actor_class) if remote_options else ray.remote(actor_class)

        return remote_class.options(
            num_cpus=self.spec.scheduling.num_cpus_per_worker,
            num_gpus=self.spec.scheduling.num_gpus_per_worker,
            **(dict(scheduling_strategy=s) if (s := scheduling_strategy) is not None else {}),
            runtime_env={"env_vars": self.spec.env_var(self.launch_context)},
            **(compute_ray_pin_head_options() if self.spec.scheduling.pin_to_head else {}),
        ).remote(**ctor_kwargs)

    async def stop(self) -> None:
        if self.actor_handle is None:
            return

        await self._shutdown_gracefully()

        try:
            ray.kill(self.actor_handle)
            logger.info(f"Killed actor at {self=}")
        except Exception as e:
            logger.warning(f"Failed to kill actor at {self=} ({e})")

    async def _shutdown_gracefully(self) -> None:
        pass

    @property
    def name(self) -> str:
        return compute_worker_name(
            pool_id=self.spec.name,
            cell_index=self.parent.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
        )

    @property
    def generation(self) -> int:
        return self.parent.generation

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
        self.actor_handle = self._create_actor(CommandActor)

    async def post_setup(self) -> None:
        ctx = LaunchCommandContext(
            **dict(self.launch_context),
            self_addrs={
                **self.self_addrs,
                **self.parent.actors[0].master_mode_addrs,
            },
            pool_addrs=self.manager.get_addrs(),
            local_gpu_ids=self.local_gpu_ids,
        )
        launch_cmd = self.spec.launch_command(ctx)
        self.actor_handle.run.remote(cmd=launch_cmd, envs={})

    async def _shutdown_gracefully(self) -> None:
        try:
            await asyncio.wait_for(self.actor_handle.shutdown.remote(), timeout=_SHUTDOWN_TIMEOUT)
        except Exception as e:
            logger.warning(f"Graceful shutdown of {self=} failed ({e})")


@dataclass
class _ServeActorManager(_BaseActorManager[ServeWorkerSpec]):
    def _compute_remote_options(self) -> dict:
        groups = self.spec.concurrency_groups
        return {} if groups is None else dict(concurrency_groups=groups)

    async def launch_actor(self) -> None:
        self.actor_handle = self._create_actor(
            self._compute_actor_class(),
            ctor_kwargs=self.spec.ctor_kwargs,
            context=self.launch_context,
        )

    def _compute_actor_class(self) -> type:
        actor_class = bootstrapped_worker_class(self.spec.worker_class)
        method_groups = self._compute_method_concurrency_groups(actor_class)
        if not method_groups:
            return actor_class
        return type(
            f"{actor_class.__name__}WithConcurrencyGroups",
            (actor_class,),
            {
                name: _route_method_to_concurrency_group(getattr(actor_class, name), group=group)
                for name, group in method_groups.items()
            },
        )

    def _compute_method_concurrency_groups(self, actor_class: type) -> dict[str, str]:
        if self.spec.concurrency_groups is None:
            return {}
        return {**declared_concurrency_groups(actor_class), **(self.spec.method_concurrency_groups or {})}

    async def post_setup(self) -> None:
        pass


def _route_method_to_concurrency_group(method: Callable, *, group: str) -> Callable:
    @functools.wraps(method)
    def routed(self, *args, **kwargs):
        return method(self, *args, **kwargs)

    return ray.method(concurrency_group=group)(routed)


@functools.cache
def bootstrapped_worker_class(worker_class_path: str) -> type:
    worker_class = load_function(worker_class_path)

    class BootstrappedWorker(worker_class, NodeProbeMixin):
        def __init__(
            self, *, ctor_kwargs: Callable[[WorkerCtorContext], dict[str, Any]], context: WorkerLaunchContext
        ) -> None:
            super().__init__(**ctor_kwargs(_ctor_context(context)))

    BootstrappedWorker.__name__ = worker_class.__name__
    BootstrappedWorker.__qualname__ = worker_class.__qualname__
    BootstrappedWorker.__module__ = worker_class.__module__
    return BootstrappedWorker


def _ctor_context(launch_context: WorkerLaunchContext) -> WorkerCtorContext:
    return WorkerCtorContext(
        cell_index=launch_context.cell_index,
        worker_in_cell_index=launch_context.worker_in_cell_index,
        gpu_ids=launch_context.gpu_ids,
        capability=DeferredBackendCapability(create=_create_ray_backend_capability),
    )


def _create_ray_backend_capability() -> BackendCapability:
    return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())


async def _gather_or_raise(coros: list[Coroutine[Any, Any, None]]) -> None:
    results = await asyncio.gather(*coros, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
