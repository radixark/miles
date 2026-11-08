"""The worker manager of a ``--cluster-backend monarch`` run, as a Monarch actor that owns every cell's proc mesh."""

from __future__ import annotations

import logging
from typing import Any

from monarch.actor import HostMesh, MeshFailure, concurrent_endpoint, endpoint, this_host

from miles.utils.test_utils.fault_injector.controller import FaultHookCommand
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, ObservedFaultHookTarget
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.monarch.base import MilesActor
from miles.utils.workers.monarch.handles import call_endpoint, get_endpoint
from miles.utils.workers.monarch.hosts import acquire_worker_hosts, check_mooncake_master_reaches_every_host
from miles.utils.workers.monarch.runtime import (
    GpuLayoutRequest,
    MonarchWorkerRuntime,
    compute_monarch_gpu_layout,
    install_manager_handle,
    installed_manager_handle,
    probe_hosts,
    read_run_env,
)
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_manager.client import ManagerBackendCapability, WorkerManagerClient
from miles.utils.workers.worker_manager.core import WorkerManager
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import BaseSpec, NamedHostAndPorts

logger = logging.getLogger(__name__)

_MANAGER_NAME = "miles-worker-manager"


class MonarchWorkerManager(MilesActor):
    """The run's worker manager: the shared WorkerManager core, running its cells through MonarchWorkerRuntime.

    It lives in a process of its own on the driver's host, and its endpoints are the verbs MonarchWorkerManagerClient
    calls, the same ones RayWorkerManager answers as a Ray actor.
    """

    def __init__(self) -> None:
        self._core = WorkerManager()
        self._runtime: MonarchWorkerRuntime | None = None

    @staticmethod
    def launch(
        args, specs: list[BaseSpec], *, comm_backend: WorkerCommBackend, gpu_layout: GpuLayoutRequest
    ) -> MonarchWorkerManager:
        """Spawn the manager in a process of its own on the driver's host, and start every cell."""
        # Attach before this_host() starts this process's runtime: loading a job's state sets the transport the
        # job's hosts reply on, and a runtime already serving on a node-local socket never hears their replies.
        worker_hosts = acquire_worker_hosts()
        if worker_hosts is not None:
            check_mooncake_master_reaches_every_host(
                args.mooncake_store_init_kwargs, num_hosts=worker_hosts.extent.nelements
            )
        head_host = this_host()
        worker_hosts = worker_hosts or head_host
        manager_procs = head_host.spawn_procs(per_host={"procs": 1}, name=_MANAGER_NAME)
        manager = manager_procs.spawn(_MANAGER_NAME, MonarchWorkerManager)
        install_manager_handle(manager)
        try:
            manager.init.call_one(
                args,
                specs,
                comm_backend=comm_backend,
                gpu_layout=gpu_layout,
                head_host=head_host,
                worker_hosts=worker_hosts,
                manager=manager,
                run_env=read_run_env(),
            ).get()
        except BaseException:
            # Nobody is handed a manager that failed to start, so nobody else would stop its process. It is a
            # BaseException so a Ctrl-C during the start stops it too; it is re-raised unchanged.
            manager_procs.stop().get()
            raise
        return manager

    @endpoint
    async def init(
        self,
        args,
        specs: list[BaseSpec],
        *,
        comm_backend: WorkerCommBackend,
        gpu_layout: GpuLayoutRequest,
        head_host: HostMesh,
        worker_hosts: HostMesh,
        manager: MonarchWorkerManager,
        run_env: dict[str, str],
    ) -> None:
        """Probe the worker hosts, lay the gpu slots out on them, and start every pool's cells.

        ``manager`` is this actor's own handle, which the runtime hands every worker it launches, since Monarch has no
        registry a worker could look the manager up in.
        """
        install_manager_handle(manager)
        hosts = await probe_hosts(worker_hosts, run_env=run_env)
        layout = compute_monarch_gpu_layout(request=gpu_layout, hosts=hosts)
        logger.info(f"Monarch hosts {hosts} lay out {gpu_layout.num_gpus} gpus as {layout}")
        self._runtime = MonarchWorkerRuntime(
            head_host=head_host,
            worker_hosts=worker_hosts,
            layout=layout,
            manager=manager,
            capability_factory=installed_monarch_backend_capability,
            run_env=run_env,
        )
        await self._core.init_pools(args, specs, runtime=self._runtime, comm_backend=comm_backend)

    @concurrent_endpoint
    async def get_cell_infos(self, *, pool_ids: list[str] | None, category: str | None) -> dict[str, CellInfo]:
        return self._core.get_cell_infos(pool_ids=pool_ids, category=category)

    @concurrent_endpoint
    async def get_worker_infos(self, cell_id: str) -> list[WorkerInfo]:
        return self._core.get_worker_infos(cell_id)

    @concurrent_endpoint
    async def get_worker_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return self._core.get_worker_addrs(worker_name)

    @concurrent_endpoint
    async def get_actor_handle(self, worker_name: str, *, expected_generation: int) -> Any:
        return self._core.get_actor_handle(worker_name, expected_generation=expected_generation)

    @concurrent_endpoint
    async def start_cells(self, cell_ids: list[str]) -> None:
        await self._core.start_cells(cell_ids)

    @concurrent_endpoint
    async def stop_cells(self, cell_ids: list[str]) -> None:
        await self._core.stop_cells(cell_ids)

    @concurrent_endpoint
    async def shutdown(self) -> None:
        await self._core.shutdown()

    @concurrent_endpoint
    async def observe_fault_target(self, cell_id: str, *, rank: int) -> ObservedFaultHookTarget:
        return self._core.observe_fault_target(cell_id, rank=rank)

    @concurrent_endpoint
    async def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord:
        return await self._core.control_fault_hook(command)

    async def __supervise__(self, failure: MeshFailure) -> bool:
        """A worker process died: keep the run alive. A cell's mesh fails every call once it lost a process, so the
        liveness scan finds the cell dead and tears it down, as under Ray."""
        logger.info(f"Mesh {failure.mesh_name} lost a process: {failure.report()}")
        return True


class MonarchWorkerManagerClient(WorkerManagerClient):
    """Every call raises the manager's own exception, as a Ray actor call would, so callers catch what it raised."""

    def __init__(self, manager: MonarchWorkerManager) -> None:
        self._manager = manager

    async def get_cell_infos(self, *, pool_ids: list[str] | None, category: str | None) -> dict[str, CellInfo]:
        return await call_endpoint(
            self._manager.get_cell_infos.call_one(pool_ids=pool_ids, category=category), describe="get_cell_infos"
        )

    async def get_worker_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return await call_endpoint(self._manager.get_worker_addrs.call_one(worker_name), describe="get_worker_addrs")

    async def start_cells(self, cell_ids: list[str]) -> None:
        await call_endpoint(self._manager.start_cells.call_one(cell_ids), describe="start_cells")

    async def stop_cells(self, cell_ids: list[str]) -> None:
        await call_endpoint(self._manager.stop_cells.call_one(cell_ids), describe="stop_cells")

    async def observe_fault_target(self, cell_id: str, *, rank: int) -> ObservedFaultHookTarget:
        return await call_endpoint(
            self._manager.observe_fault_target.call_one(cell_id, rank=rank), describe="observe_fault_target"
        )

    async def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord:
        return await call_endpoint(self._manager.control_fault_hook.call_one(command), describe="control_fault_hook")

    def get_worker_infos_blocking(self, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        # every request goes out before the first answer is awaited, as one ray.get over many refs would
        futures = [self._manager.get_worker_infos.call_one(cell_id) for cell_id in cell_ids]
        return [get_endpoint(future, describe="get_worker_infos") for future in futures]

    def get_native_handles_blocking(self, infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
        futures = {
            info.name: self._manager.get_actor_handle.call_one(info.name, expected_generation=info.generation)
            for info in infos
        }
        handles = {name: get_endpoint(future, describe="get_actor_handle") for name, future in futures.items()}
        not_served = sorted(name for name, handle in handles.items() if not isinstance(handle, BaseWorkerHandle))
        assert not not_served, f"{not_served} are launched as commands, which have no methods to call"
        return handles


def create_monarch_backend_capability(manager: MonarchWorkerManager) -> BackendCapability:
    """The backend capability that reaches the run's workers through this manager."""
    return ManagerBackendCapability(MonarchWorkerManagerClient(manager))


def installed_monarch_backend_capability() -> BackendCapability:
    """The capability of the manager installed in this process, as a launched worker or the driver reaches it."""
    return create_monarch_backend_capability(installed_manager_handle())


def launch_monarch_worker_manager(args: Any, specs: list[BaseSpec]) -> MonarchWorkerManager:
    """The driver's entry point: launch the manager with the run's gpu layout, and return its handle once every cell
    started."""
    # Lay out the gpu slots with the Ray placement groups' own math instead of a copy of it: reusing it is
    # convenient, and one copy keeps a run's gpu layout from diverging between the two backends.
    from miles.ray.placement_group import _get_placement_group_layout

    num_gpus, rollout_offset = _get_placement_group_layout(args)
    return MonarchWorkerManager.launch(
        args,
        specs,
        comm_backend=WorkerCommBackend(args.worker_comm_backend),
        gpu_layout=GpuLayoutRequest(
            num_gpus=num_gpus,
            rollout_offset=rollout_offset,
            use_critic=args.use_critic,
            num_gpus_per_node=args.num_gpus_per_node,
        ),
    )


async def shutdown_monarch_worker_manager(manager: MonarchWorkerManager) -> None:
    await call_endpoint(manager.shutdown.call_one(), describe="shutdown")
