from __future__ import annotations

import asyncio

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_provider.base import CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesWorkerProvider


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

    async def inject_fault(self, *, cell_id: str, mode: FailureMode, sub_index: int) -> None:
        raise NotImplementedError("fault injection reaches into a worker process, which needs the rpc layer")

    async def _ensure_watching(self) -> None:
        if self._watching is None:
            self._watching = asyncio.ensure_future(self._provider.watch_cells(_ignore_cell))
        try:
            await asyncio.shield(self._watching)
        except BaseException:
            self._watching = None
            raise


async def _ignore_cell(cell_id: str, info: CellInfo | None) -> None:
    return None


async def _delete_pods(*, namespace: str, pod_names: list[str]) -> None:
    from kubernetes_asyncio import client as kubernetes_client
    from kubernetes_asyncio import config as kubernetes_config

    kubernetes_config.load_incluster_config()
    async with kubernetes_client.ApiClient() as api_client:
        core_v1_api = kubernetes_client.CoreV1Api(api_client)
        await asyncio.gather(
            *(core_v1_api.delete_namespaced_pod(name=pod_name, namespace=namespace) for pod_name in pod_names)
        )
