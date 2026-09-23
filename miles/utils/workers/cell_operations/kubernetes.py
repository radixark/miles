from __future__ import annotations

import asyncio
import logging

from miles.utils.test_utils.fault_injector.actions.process import FailureMode
from miles.utils.test_utils.fault_injector.controller import FaultHookCommand
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, ObservedFaultHookTarget
from miles.utils.workers.cell_operations.base import BaseCellOperations, StaleFaultTargetError
from miles.utils.workers.k8s_client import core_v1_api
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError
from miles.utils.workers.worker_provider.base import CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesWorkerProvider
from miles.utils.workers.worker_provider.utils import build_rpc_handle_of_worker_info

logger = logging.getLogger(__name__)

INJECT_FAULT_TIMEOUT_SECONDS = 60.0
CONTROL_FAULT_HOOK_TIMEOUT_SECONDS = 10.0


class KubernetesCellOperations(BaseCellOperations):
    def __init__(self, *, provider: KubernetesWorkerProvider, namespace: str) -> None:
        self._provider = provider
        self._namespace = namespace
        self._watching: asyncio.Task[StopWatchFn] | None = None

    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        await self._ensure_watching()

        wanted = set(pool_ids)
        infos = (self._provider.cell_info(cell_id) for cell_id in self._provider.cell_ids())
        return {info.cell_id: info for info in infos if info is not None and info.pool_id in wanted}

    async def suspend(self, *, cell_id: str) -> None:
        await self._ensure_watching()

        pods = self._provider.pod_names_of_cell(cell_id)
        assert pods, f"cannot suspend {cell_id}, which has no pods"
        await _delete_pods(namespace=self._namespace, pod_names=pods)

    async def resume(self, *, cell_id: str) -> None:
        raise NotImplementedError(
            "a deleted cell comes back when its workload recreates it, so resume has no moment to return at"
        )

    async def observe_fault_target(self, *, cell_id: str, rank: int) -> ObservedFaultHookTarget:
        await self._ensure_watching()

        (infos,) = self._provider.get_worker_infos(cell_ids=[cell_id])
        if not 0 <= rank < len(infos):
            raise StaleFaultTargetError(f"Cell {cell_id} has no worker at index {rank}")
        info = infos[rank]
        if info.worker_class is None:
            raise NotImplementedError(f"Worker {info.name} is not served over RPC")

        health = await build_rpc_handle_of_worker_info(info).read_health()
        if not health.boot_uuid or not health.pod_uid:
            raise StaleFaultTargetError(f"Worker {info.name} reports no boot or pod identity")

        if (incarnation := self._provider.debug_cell_incarnation(cell_id)) is None:
            raise StaleFaultTargetError(f"Cell {cell_id} has disappeared")
        if health.pod_uid not in {pod.uid for pod in incarnation.pods}:
            raise StaleFaultTargetError(f"Worker {info.name} answered from a pod that cell {cell_id} no longer lists")
        return ObservedFaultHookTarget(
            cell_id=cell_id,
            rank=rank,
            workers_hash=incarnation.workers_hash,
            boot_uuid=health.boot_uuid,
            pod_uid=health.pod_uid,
        )

    async def inject_fault(
        self,
        *,
        cell_id: str,
        mode: FailureMode,
        sub_index: int,
        expected_target: ObservedFaultHookTarget | None = None,
    ) -> None:
        await self._ensure_watching()

        if expected_target is not None and expected_target != await self.observe_fault_target(
            cell_id=cell_id, rank=sub_index
        ):
            raise StaleFaultTargetError(f"Cell {cell_id} no longer matches the observed fault target")

        (infos,) = self._provider.get_worker_infos(cell_ids=[cell_id])
        assert (
            0 <= sub_index < len(infos)
        ), f"sub_index {sub_index} is out of range for cell {cell_id}, which has {len(infos)} workers"

        info = infos[sub_index]
        handle = build_rpc_handle_of_worker_info(
            info, expected_boot_uuid=expected_target.boot_uuid if expected_target is not None else None
        )
        await _inject_fault_over_rpc(handle=handle, mode=mode, worker_name=info.name)

    async def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord:
        await self._ensure_watching()

        target = command.request.target
        assert isinstance(target, ObservedFaultHookTarget), "A fault hook sent to a cell names the worker it observed"
        if target != await self.observe_fault_target(cell_id=target.cell_id, rank=target.rank):
            raise StaleFaultTargetError(f"Cell {target.cell_id} no longer matches the observed fault target")
        (infos,) = self._provider.get_worker_infos(cell_ids=[target.cell_id])
        handle = build_rpc_handle_of_worker_info(infos[target.rank], expected_boot_uuid=target.boot_uuid)
        try:
            return await asyncio.wait_for(
                handle.control_fault_hook(command=command), timeout=CONTROL_FAULT_HOOK_TIMEOUT_SECONDS
            )
        except ServerRestartedError as error:
            raise StaleFaultTargetError("Fault hook worker changed its boot identity") from error

    async def _ensure_watching(self) -> None:
        if self._watching is None:
            self._watching = asyncio.ensure_future(self._provider.watch_cells(_ignore_cell))
        try:
            await self._watching
        except BaseException:
            self._watching = None
            raise


async def _inject_fault_over_rpc(*, handle: BaseWorkerHandle, mode: FailureMode, worker_name: str) -> None:
    try:
        await asyncio.wait_for(
            handle.submit_without_result("inject_fault", mode=mode.value), timeout=INJECT_FAULT_TIMEOUT_SECONDS
        )
    except ServerRestartedError as error:
        raise StaleFaultTargetError(f"Worker {worker_name} changed its boot identity") from error
    except (WorkerUnreachableError, TimeoutError, asyncio.TimeoutError):
        logger.info("Injecting %s into %s left it unreachable, which is what was asked for", mode.value, worker_name)


async def _ignore_cell(cell_id: str, info: CellInfo | None) -> None:
    return None


async def _delete_pods(*, namespace: str, pod_names: list[str]) -> None:
    async with core_v1_api() as api:
        await asyncio.gather(*(api.delete_namespaced_pod(name=pod_name, namespace=namespace) for pod_name in pod_names))
