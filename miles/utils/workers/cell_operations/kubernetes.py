from __future__ import annotations

from collections.abc import Callable

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_provider.base import CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.kubernetes.client import delete_pods
from miles.utils.workers.worker_provider.kubernetes.provider import KubernetesWorkerProvider


class KubernetesCellOperations(BaseCellOperations):
    def __init__(
        self,
        *,
        provider: KubernetesWorkerProvider,
        namespace: str,
        colocated_with: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._provider = provider
        self._namespace = namespace
        self._colocated_with = colocated_with
        self._stop_watch: StopWatchFn | None = None

    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        await self._ensure_watching()

        wanted = set(pool_ids)
        infos = (self._provider.cell_info(cell_id) for cell_id in self._provider.cell_ids())
        return {info.cell_id: info for info in infos if info is not None and info.pool_id in wanted}

    async def suspend(self, *, cell_id: str) -> None:
        await self._ensure_watching()

        pods = self._provider.pod_names(cell_id)
        assert pods, f"cannot suspend {cell_id}, which has no pods"

        if self._colocated_with is not None:
            pods = pods + self._colocated_with(cell_id)
        await delete_pods(namespace=self._namespace, pod_names=pods)

    async def resume(self, *, cell_id: str) -> None:
        raise NotImplementedError(
            "a deleted cell comes back when its workload recreates it, so resume has no moment to return at"
        )

    async def inject_fault(self, *, cell_id: str, mode: FailureMode, sub_index: int) -> None:
        raise NotImplementedError("fault injection reaches into a worker process, which needs the rpc layer")

    async def _ensure_watching(self) -> None:
        if self._stop_watch is None:
            self._stop_watch = await self._provider.watch_cells(_ignore_cell)


async def _ignore_cell(cell_id: str, info: CellInfo | None) -> None:
    return None
