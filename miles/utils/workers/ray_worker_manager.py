from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.ray_utils import compute_ray_pin_head_options
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.rpc.common.metadata import declared_concurrency_groups
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_manager.core import WorkerManager
from miles.utils.workers.worker_manager.runtime import GpuSlotRef, WorkerLaunchRequest, WorkerProcess, WorkerRuntime
from miles.utils.workers.worker_spec import BaseSpec

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from miles.ray.placement_group import PlacementGroupInfo

# TODO: unique name, maybe with args.run_uuid
_ACTOR_NAME = "ray_worker_manager"


class RayWorkerManager(WorkerManager):
    @staticmethod
    def launch(args, specs: list[BaseSpec], pgs: dict[str, PlacementGroupInfo], *, comm_backend: WorkerCommBackend):
        obj = ray.remote(RayWorkerManager).options(name=_ACTOR_NAME).remote()
        ray.get(obj.init.remote(args, specs, pgs, comm_backend=comm_backend))
        return obj

    @staticmethod
    def get_handle() -> ray.actor.ActorHandle:
        return ray.get_actor(_ACTOR_NAME)

    async def init(
        self, args, specs: list[BaseSpec], pgs: dict[str, PlacementGroupInfo], *, comm_backend: WorkerCommBackend
    ):
        await self.init_pools(args, specs, runtime=RayWorkerRuntime(pgs=pgs), comm_backend=comm_backend)


class RayWorkerRuntime(WorkerRuntime):
    def __init__(self, *, pgs: dict[str, PlacementGroupInfo]) -> None:
        self._pgs = pgs

    async def launch_cell(self, *, cell_id: str, launches: list[WorkerLaunchRequest]) -> list[WorkerProcess]:
        processes: list[WorkerProcess] = []
        try:
            for launch in launches:
                processes.append(_RayWorkerProcess(self._create_actor(launch)))
        except BaseException:
            # The core never learns of processes it was not handed, so nobody else would kill these.
            for process in processes:
                await self.kill_worker(process)
            raise
        return processes

    async def kill_worker(self, process: WorkerProcess) -> None:
        try:
            ray.kill(process.native_handle)
            logger.info(f"Killed actor {process.native_handle}")
        except Exception as e:
            logger.warning(f"Failed to kill actor {process.native_handle} ({e})")

    async def release_cell(self, *, processes: list[WorkerProcess]) -> None:
        pass

    def physical_gpu_ids(self, slots: GpuSlotRef) -> list[int]:
        base_gpu_id = int(self._pgs[slots.layout_name].pg_reordered_gpu_ids[slots.first_slot_index])
        return list(range(base_gpu_id, base_gpu_id + slots.num_slots))

    def worker_capability_factory(self) -> Callable[[], BackendCapability]:
        return _create_ray_backend_capability

    def _create_actor(self, launch: WorkerLaunchRequest) -> ray.actor.ActorHandle:
        scheduling_strategy = None
        if (slots := launch.gpu_slots) is not None:
            pg = self._pgs[slots.layout_name]
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg.pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=pg.pg_reordered_bundle_indices[slots.first_slot_index],
            )

        actor_class = _compute_actor_class(launch)
        remote_options = (
            {} if launch.concurrency_groups is None else dict(concurrency_groups=launch.concurrency_groups)
        )
        remote_class = ray.remote(**remote_options)(actor_class) if remote_options else ray.remote(actor_class)

        return remote_class.options(
            num_cpus=launch.num_cpus,
            num_gpus=launch.num_gpus,
            **(dict(scheduling_strategy=s) if (s := scheduling_strategy) is not None else {}),
            runtime_env={"env_vars": launch.env_vars},
            **(compute_ray_pin_head_options() if launch.pin_to_head else {}),
        ).remote(**launch.ctor_kwargs)


class _RayWorkerProcess(WorkerProcess):
    def __init__(self, actor_handle: ray.actor.ActorHandle) -> None:
        self._actor_handle = actor_handle

    async def call(self, method: str, /, **kwargs: Any) -> Any:
        return await getattr(self._actor_handle, method).remote(
            **kwargs
        )  # config-access-exempt: attribute selected at runtime from method

    def call_without_result(self, method: str, /, **kwargs: Any) -> None:
        getattr(self._actor_handle, method).remote(**kwargs)  # config-access-exempt: attribute selected at runtime

    async def probe_is_dead(self) -> bool:
        return await RayWorkerHandle(self._actor_handle).probe_is_dead()

    @property
    def native_handle(self) -> ray.actor.ActorHandle:
        return self._actor_handle


def _compute_actor_class(launch: WorkerLaunchRequest) -> type:
    actor_class = launch.actor_class
    if (groups := launch.concurrency_groups) is None:
        return actor_class

    method_groups = declared_concurrency_groups(actor_class)
    assert method_groups, (
        f"Worker {launch.pool_id!r} declares concurrency groups {sorted(groups)} but no method of "
        f"{actor_class.__name__} is annotated with @rpc(concurrency_group=...): threading the actor "
        f"while every method stays in the default group buys nothing"
    )
    undeclared = sorted(set(method_groups.values()) - set(groups))
    assert not undeclared, f"Worker {launch.pool_id!r} routes methods to undeclared groups: {undeclared}"
    return type(
        f"{actor_class.__name__}WithConcurrencyGroups",
        (actor_class,),
        {
            name: _route_method_to_concurrency_group(
                getattr(actor_class, name), group=group
            )  # config-access-exempt: attribute selected at runtime from name
            for name, group in method_groups.items()
        },
    )


def _route_method_to_concurrency_group(method: Callable, *, group: str) -> Callable:
    @functools.wraps(method)
    def routed(self, *args, **kwargs):
        return method(self, *args, **kwargs)

    routed.__signature__ = inspect.signature(method)
    del routed.__wrapped__
    return ray.method(concurrency_group=group)(routed)


def _create_ray_backend_capability() -> BackendCapability:
    return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())


def launch_ray_worker_manager(args: Any, specs: list[BaseSpec]) -> ray.actor.ActorHandle:
    # placement_group reaches the capability through this module, so a top-level import back would deadlock both
    from miles.ray.placement_group import create_placement_groups

    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(args, specs, pgs, comm_backend=WorkerCommBackend(args.worker_comm_backend))


async def shutdown_ray_worker_manager(handle: ray.actor.ActorHandle) -> None:
    try:
        await handle.shutdown.remote()
    finally:
        ray.kill(handle)
