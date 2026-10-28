import ray.actor

from miles.utils.test_utils.fault_injector.controller import FaultHookCommand
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, ObservedFaultHookTarget
from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_manager.client import WorkerManagerClient
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import NamedHostAndPorts


class RayWorkerManagerClient(WorkerManagerClient):
    def __init__(self, worker_manager_handle: ray.actor.ActorHandle) -> None:
        self._handle = worker_manager_handle

    async def get_cell_infos(self, *, pool_ids: list[str] | None, category: str | None) -> dict[str, CellInfo]:
        return await self._handle.get_cell_infos.remote(pool_ids=pool_ids, category=category)

    async def get_worker_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return await self._handle.get_worker_addrs.remote(worker_name)

    async def start_cells(self, cell_ids: list[str]) -> None:
        await self._handle.start_cells.remote(cell_ids)

    async def stop_cells(self, cell_ids: list[str]) -> None:
        await self._handle.stop_cells.remote(cell_ids)

    async def observe_fault_target(self, cell_id: str, *, rank: int) -> ObservedFaultHookTarget:
        return await self._handle.observe_fault_target.remote(cell_id, rank=rank)

    async def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord:
        return await self._handle.control_fault_hook.remote(command=command)

    def get_worker_infos_blocking(self, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return ray.get([self._handle.get_worker_infos.remote(cell_id) for cell_id in cell_ids])

    def get_native_handles_blocking(self, infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
        refs = [self._handle.get_actor_handle.remote(info.name, expected_generation=info.generation) for info in infos]
        return {info.name: RayWorkerHandle(handle) for info, handle in zip(infos, ray.get(refs), strict=True)}
