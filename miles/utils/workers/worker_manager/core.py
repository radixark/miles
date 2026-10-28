"""The backend-neutral worker manager: pools, cells, generations, ports and liveness.

How a cell's processes are started, called and stopped is the job of the ``WorkerRuntime`` it is built with.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.function_registry import load_function
from miles.utils.http_utils import wrap_ipv6
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import NodeProbeMixin
from miles.utils.test_utils.fault_injector.controller import FaultHookCommand
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, ObservedFaultHookTarget
from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.backend_capability.base import BackendCapability, DeferredBackendCapability
from miles.utils.workers.cell_operations.base import StaleFaultTargetError
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.serving.serve_actor import ServeActor
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_manager.runtime import GpuSlotRef, WorkerLaunchRequest, WorkerProcess, WorkerRuntime
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    BaseCommandSpec,
    BaseServeSpec,
    BaseSpec,
    HostAndPort,
    LaunchCommandContext,
    NamedHostAndPorts,
    WorkerCtorContext,
    WorkerLaunchContext,
)

logger = logging.getLogger(__name__)

_LIVENESS_SCAN_INTERVAL_SECONDS = 10.0


class WorkerManager:
    runtime: WorkerRuntime

    def __init__(self):
        self.port_allocator = PortAllocator()

    async def init_pools(
        self, args, specs: list[BaseSpec], *, runtime: WorkerRuntime, comm_backend: WorkerCommBackend
    ) -> None:
        configure_logger(args, source=SimpleProcessIdentity(component="worker_manager"))

        self.scaling = ScalingConfig.slice_from(args)
        self.comm_backend = comm_backend
        self.runtime = runtime
        self._pools = {spec.name: _PoolManager.initial(spec, self) for spec in specs}
        assert len(self._pools) == len(specs)
        self._membership_lock = asyncio.Lock()

        await self.start_cells([c.cell_id for c in self._all_cells()])

    async def start_cells(self, cell_ids: list[str]) -> None:
        async with self._membership_lock:
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
        async with self._membership_lock:
            await asyncio.gather(*[self._find_cell(cell_id).stop() for cell_id in cell_ids])

    async def shutdown(self) -> None:
        async with self._membership_lock:
            await asyncio.gather(*[cell.stop() for cell in self._all_cells()])

    def observe_fault_target(self, cell_id: str, *, rank: int) -> ObservedFaultHookTarget:
        cell = self._find_cell(cell_id)
        if not cell.alive or not 0 <= rank < len(cell.actors):
            raise StaleFaultTargetError(f"Cell {cell_id} has no live worker at index {rank}")
        return ObservedFaultHookTarget(cell_id=cell_id, rank=rank, workers_hash=cell.get_info().workers_hash)

    async def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord:
        target = command.request.target
        assert isinstance(target, ObservedFaultHookTarget), "A fault hook sent to a cell names the worker it observed"
        if self.observe_fault_target(cell_id=target.cell_id, rank=target.rank) != target:
            raise StaleFaultTargetError("Fault hook target no longer matches the observed worker")
        process = self._find_cell(target.cell_id).actors[target.rank].process
        return await process.call("control_fault_hook", command=command)

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

    def get_cell_infos(self, *, pool_ids: list[str] | None, category: str | None = None) -> dict[str, CellInfo]:
        # TODO: about `get_worker_infos` (which is only used by dashboard)
        if pool_ids is None:
            pool_ids = [
                name
                for name, pool in self._pools.items()
                if pool.spec.scheduling(self.scaling).declares_dynamic_pool()
            ]
        unknown = set(pool_ids) - set(self._pools)
        assert not unknown, f"{unknown=} {sorted(self._pools)=}"
        infos = [
            c.get_info()
            for name in pool_ids
            if category is None or self._pools[name].spec.category == category
            for c in self._pools[name].cells
        ]
        return {info.cell_id: info for info in infos}

    def get_actor_handle(self, worker_name: str, *, expected_generation: int) -> Any:
        actor = self._find_actor(worker_name)
        assert actor.generation == expected_generation, (
            f"{worker_name} is now generation {actor.generation}, not the {expected_generation} it was described as; "
            f"ask for its worker infos again"
        )
        return actor.process.native_handle

    def _compute_worker_info(self, actor: _BaseActorManager) -> WorkerInfo:
        served_over_rpc = isinstance(actor.spec, BaseServeSpec) and self.comm_backend == WorkerCommBackend.RPC
        return WorkerInfo(
            name=actor.name,
            generation=actor.generation,
            self_addrs=actor.self_addrs or {},
            gpu_ids=actor.gpu_ids,
            worker_class=actor.spec.worker_class if served_over_rpc else None,
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
    spec: BaseSpec
    cells: list[_CellManager]

    @classmethod
    def initial(cls, spec: BaseSpec, manager: WorkerManager) -> _PoolManager:
        return cls(
            spec=spec,
            cells=[
                _CellManager(
                    manager=manager,
                    cell_index=cell_index,
                    spec=spec,
                    actors=None,
                )
                for cell_index in range(spec.scheduling(manager.scaling).num_cells)
            ],
        )


SpecT = TypeVar("SpecT", bound=BaseSpec)


def _actor_manager_cls(spec: BaseSpec, *, comm_backend: WorkerCommBackend) -> type[_BaseActorManager]:
    match spec, comm_backend:
        case BaseCommandSpec(), _:
            return _CommandActorManager
        case BaseServeSpec(), WorkerCommBackend.RPC:
            return _ServeActorRpcCommManager
        case BaseServeSpec(), backend if backend.calls_workers_natively():
            return _ServeActorNativeCommManager
    raise AssertionError(f"{spec.name} is neither served nor launched as a command")


@dataclass(kw_only=True)
class _CellManager(Generic[SpecT]):
    manager: WorkerManager
    cell_index: int
    spec: SpecT
    actors: list[_BaseActorManager] | None
    generation: int = 0
    liveness_scan_task: asyncio.Task | None = None

    async def launch_actors(self):
        assert self.actors is None
        self.generation += 1
        scheduling = self.spec.scheduling(self.manager.scaling)
        actor_manager_cls = _actor_manager_cls(self.spec, comm_backend=self.manager.comm_backend)
        self.actors = [
            actor_manager_cls(
                manager=self.manager,
                parent=self,
                worker_in_cell_index=worker_in_cell_index,
                spec=self.spec,
                process=None,
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
        processes = await self.manager.runtime.launch_cell(
            cell_id=self.cell_id, launches=[a.compute_launch() for a in self.actors]
        )
        for actor, process in zip(self.actors, processes, strict=True):
            actor.process = process
        self.liveness_scan_task = asyncio.create_task(self._scan_liveness_forever(self.generation))

    async def alloc_ports(self) -> None:
        await self._for_all_actors(lambda a: a.alloc_ports())

    async def post_setup(self) -> None:
        await self._for_all_actors(lambda a: a.post_setup())

    async def stop(self) -> None:
        if self.actors is None:
            return
        await self._for_all_actors(lambda a: a.stop())
        # a cell whose launch failed holds no processes: launch_cell cleaned up the ones it had created
        if processes := [a.process for a in self.actors if a.process is not None]:
            await self.manager.runtime.release_cell(processes=processes)
        self.actors = None

    async def _scan_liveness_forever(self, generation: int) -> None:
        while self.generation == generation and self.actors is not None:
            await asyncio.sleep(_LIVENESS_SCAN_INTERVAL_SECONDS)
            try:
                await self._scan_liveness_once()
            except Exception:
                logger.error(f"Scanning liveness of cell {self.cell_id} failed, will scan again", exc_info=True)

    async def _scan_liveness_once(self) -> None:
        generation = self.generation
        dead_worker_names = await self._find_dead_worker_names()
        if not dead_worker_names:
            return

        async with self.manager._membership_lock:
            if self.actors is None or self.generation != generation:
                return
            logger.error(
                f"Cell {self.cell_id} lost workers {dead_worker_names} without being stopped, "
                f"so the whole cell is torn down and reported as not alive"
            )
            await self.stop()

    async def _find_dead_worker_names(self) -> list[str]:
        if (actors := self.actors) is None:
            return []
        probes = await asyncio.gather(*[a.probe_is_dead() for a in actors])
        return [actor.name for actor, is_dead in zip(actors, probes, strict=True) if is_dead]

    async def _for_all_actors(self, fn: Callable[[_BaseActorManager], Any]):
        await asyncio.gather(*[fn(a) for a in self.actors])

    def get_info(self) -> CellInfo:
        return CellInfo(
            cell_id=self.cell_id,
            pool_id=self.spec.name,
            alive=self.alive and self._all_workers_have_addrs,
            worker_names=[a.name for a in self.actors] if self.actors is not None else [],
            workers_hash=f"pseudo-hash-{self.generation}",
            meta=self.spec.compute_static_meta(self.manager.scaling).resolve(cell_index=self.cell_index),
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
    manager: WorkerManager
    parent: _CellManager
    worker_in_cell_index: int
    spec: SpecT
    process: WorkerProcess | None
    self_addrs: NamedHostAndPorts | None = None
    local_gpu_ids: list[int] | None = None
    gpu_slot_index: int | None

    def compute_launch(self) -> WorkerLaunchRequest:
        raise NotImplementedError

    async def post_setup(self) -> None:
        raise NotImplementedError

    async def alloc_ports(self) -> None:
        allocated: NamedHostAndPorts = {}

        node_ip, external_ip, self.local_gpu_ids = await asyncio.gather(
            self.process.call("_get_node_ip"),
            self.process.call("_get_node_external_ip"),
            self.process.call("_to_local_gpu_ids", gpu_ids=self.gpu_ids),
        )
        for port_info in self.spec.port_infos:
            if self.worker_in_cell_index != 0 and port_info.mode == "master":
                continue
            if port_info.allow_dynamic:
                port = await self.manager.port_allocator.alloc(
                    partial(self.process.call, "_get_free_port_block"),
                    node_ip=node_ip,
                    consecutive=port_info.num_consecutive,
                )
            else:
                port = port_info.static_port + (self.parent.cell_index if port_info.offset_by_cell else 0)
                await self._assert_static_port_is_free(port=port, port_name=port_info.name, node_ip=node_ip)
            allocated[port_info.name] = HostAndPort(
                host=wrap_ipv6(node_ip),
                port=port,
                external_host=wrap_ipv6(external_ip) if external_ip else None,
            )

        self.self_addrs = allocated

    async def _assert_static_port_is_free(self, *, port: int, port_name: str, node_ip: str) -> None:
        free = await self.process.call("_is_port_available", port=port)
        assert free, (
            f"Port {port} on {node_ip} is already in use, so {self.name} cannot serve its {port_name!r} "
            f"endpoint there; a stale process from an earlier run is the usual cause"
        )

    @property
    def launch_context(self) -> WorkerLaunchContext:
        return WorkerLaunchContext(
            args=self.spec.args,
            cell_index=self.parent.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
            num_workers_per_cell=self.spec.scheduling(self.manager.scaling).num_workers_per_cell,
            gpu_ids=self.gpu_ids,
        )

    def _compute_launch_of(
        self,
        actor_class: type,
        *,
        served_natively: bool = False,
        concurrency_groups: dict[str, int] | None = None,
        **ctor_kwargs,
    ) -> WorkerLaunchRequest:
        scheduling = self.spec.scheduling(self.manager.scaling)
        return WorkerLaunchRequest(
            pool_id=self.spec.name,
            name=self.name,
            actor_class=actor_class,
            ctor_kwargs=ctor_kwargs,
            env_vars=self.spec.env_var(self.launch_context),
            num_cpus=scheduling.num_cpus_per_worker,
            num_gpus=scheduling.num_gpus_per_worker,
            gpu_slots=self._gpu_slots,
            pin_to_head=scheduling.pin_to_head,
            served_natively=served_natively,
            concurrency_groups=concurrency_groups,
        )

    async def probe_is_dead(self) -> bool:
        if self.process is None:
            return False
        return await self.process.probe_is_dead()

    async def stop(self) -> None:
        if self.process is None:
            return
        await self.shutdown_gracefully()
        await self.manager.runtime.kill_worker(self.process)

    async def shutdown_gracefully(self) -> None:
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
    def _gpu_slots(self) -> GpuSlotRef | None:
        scheduling = self.spec.scheduling(self.manager.scaling)
        if (pg_name := scheduling.pg_name) is None:
            return None
        return GpuSlotRef(
            layout_name=pg_name, first_slot_index=self.gpu_slot_index, num_slots=scheduling.num_gpu_slots_per_worker
        )

    @property
    def gpu_ids(self) -> list[int]:
        if (slots := self._gpu_slots) is None:
            return []
        return self.manager.runtime.physical_gpu_ids(slots)

    @property
    def master_mode_addrs(self) -> NamedHostAndPorts:
        return {info.name: self.self_addrs[info.name] for info in self.spec.port_infos if info.mode == "master"}


@dataclass
class _CommandActorManager(_BaseActorManager[BaseCommandSpec]):
    def compute_launch(self) -> WorkerLaunchRequest:
        return self._compute_launch_of(CommandActor)

    async def post_setup(self) -> None:
        ctx = LaunchCommandContext(
            **dict(self.launch_context),
            scaling=self.manager.scaling,
            self_addrs={
                **self.self_addrs,
                **self.parent.actors[0].master_mode_addrs,
            },
            pool_addrs=self.manager.get_addrs(),
            local_gpu_ids=self.local_gpu_ids,
        )
        launch_cmd = self.spec.launch_command(ctx)
        # exec lets the command replace /bin/sh, so parent death and the runtime's child cleanup kill the
        # command itself rather than a shell that would leave it orphaned with its ports.
        self.process.call_without_result("run", cmd=f"exec {launch_cmd}", envs={})

    async def shutdown_gracefully(self) -> None:
        try:
            await asyncio.wait_for(self.process.call("shutdown"), timeout=_SHUTDOWN_TIMEOUT)
        except Exception as e:
            logger.warning(f"Graceful shutdown of {self=} failed ({e})")


@dataclass
class _ServeActorNativeCommManager(_BaseActorManager[BaseServeSpec]):
    """A served worker the driver calls through the runtime's own actor calls rather than rpc."""

    def compute_launch(self) -> WorkerLaunchRequest:
        return self._compute_launch_of(
            bootstrapped_worker_class(self.spec.worker_class),
            served_natively=True,
            concurrency_groups=self.spec.concurrency_groups,
            ctor_kwargs=self.spec.ctor_kwargs,
            context=self.launch_context,
            capability_factory=self.manager.runtime.worker_capability_factory(),
        )

    async def post_setup(self) -> None:
        pass


@dataclass
class _ServeActorRpcCommManager(_BaseActorManager[BaseServeSpec]):
    def compute_launch(self) -> WorkerLaunchRequest:
        return self._compute_launch_of(
            ServeActor,
            build_worker=partial(
                _build_serve_worker,
                worker_class_path=self.spec.worker_class,
                ctor_kwargs=self.spec.ctor_kwargs,
                context=self.launch_context,
                capability_factory=self.manager.runtime.worker_capability_factory(),
            ),
        )

    async def post_setup(self) -> None:
        await self.process.call("start_rpc_server", port=self.self_addrs[RPC_PORT_NAME].port)


def _build_serve_worker(
    *,
    worker_class_path: str,
    ctor_kwargs: Callable[[WorkerCtorContext], dict[str, Any]],
    context: WorkerLaunchContext,
    capability_factory: Callable[[], BackendCapability],
) -> Any:
    return bootstrapped_worker_class(worker_class_path)(
        ctor_kwargs=ctor_kwargs, context=context, capability_factory=capability_factory
    )


@functools.cache
def bootstrapped_worker_class(worker_class_path: str) -> type:
    worker_class = load_function(worker_class_path)

    class BootstrappedWorker(worker_class, NodeProbeMixin):
        def __init__(
            self,
            *,
            ctor_kwargs: Callable[[WorkerCtorContext], dict[str, Any]],
            context: WorkerLaunchContext,
            capability_factory: Callable[[], BackendCapability],
        ) -> None:
            super().__init__(**ctor_kwargs(_ctor_context(context, capability_factory=capability_factory)))

    BootstrappedWorker.__name__ = worker_class.__name__
    BootstrappedWorker.__qualname__ = worker_class.__qualname__
    BootstrappedWorker.__module__ = worker_class.__module__
    return BootstrappedWorker


def _ctor_context(
    launch_context: WorkerLaunchContext, *, capability_factory: Callable[[], BackendCapability]
) -> WorkerCtorContext:
    return WorkerCtorContext(
        args=launch_context.args,
        cell_index=launch_context.cell_index,
        worker_in_cell_index=launch_context.worker_in_cell_index,
        num_workers_per_cell=launch_context.num_workers_per_cell,
        gpu_ids=launch_context.gpu_ids,
        capability=DeferredBackendCapability(create=capability_factory),
    )


async def _gather_or_raise(coros: list[Coroutine[Any, Any, None]]) -> None:
    results = await asyncio.gather(*coros, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
