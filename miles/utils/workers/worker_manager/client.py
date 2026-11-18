"""How drivers and workers reach a worker manager, whichever cluster backend hosts it.

``ManagerWorkerProvider`` and ``ManagerCellOperations`` answer the backend-neutral interfaces from the manager's
bookkeeping; a backend only supplies the ``WorkerManagerClient`` that carries each verb to the manager.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Sequence
from functools import partial

from miles.utils.test_utils.fault_injector.controller import FaultHookCommand
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, ObservedFaultHookTarget
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.polling_reconcile_loop import PollingReconcileLoop
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.utils import build_rpc_handle_of_worker_info
from miles.utils.workers.worker_spec import NamedHostAndPorts

POLL_INTERVAL_SECONDS = 5.0
CONTROL_FAULT_HOOK_TIMEOUT_SECONDS = 10.0


class WorkerManagerClient(abc.ABC):
    """The verbs of ``WorkerManager`` a driver or worker calls, carried by the backend's own messaging."""

    @abc.abstractmethod
    async def get_cell_infos(self, *, pool_ids: list[str] | None, category: str | None) -> dict[str, CellInfo]: ...

    @abc.abstractmethod
    async def get_worker_addrs(self, worker_name: str) -> NamedHostAndPorts: ...

    @abc.abstractmethod
    async def start_cells(self, cell_ids: list[str]) -> None: ...

    @abc.abstractmethod
    async def stop_cells(self, cell_ids: list[str]) -> None: ...

    @abc.abstractmethod
    async def observe_fault_target(self, cell_id: str, *, rank: int) -> ObservedFaultHookTarget: ...

    @abc.abstractmethod
    async def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord: ...

    # BaseWorkerProvider answers these synchronously, so the client does too.
    @abc.abstractmethod
    def get_worker_infos_blocking(self, cell_ids: list[str]) -> list[list[WorkerInfo]]: ...

    @abc.abstractmethod
    def get_native_handles_blocking(self, infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
        """Handles of workers the driver calls through the backend's own actor calls rather than rpc."""


class ManagerWorkerProvider(BaseWorkerProvider):
    def __init__(
        self,
        client: WorkerManagerClient,
        *,
        pool_ids: list[str] | None = None,
        category: str | None = None,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    ):
        self._client = client
        self._pool_ids = pool_ids
        self._category = category
        self._poll_interval_seconds = poll_interval_seconds

    # TEMPORARY: this layer is not meant to serve a suspend the inference controller drives, deliberately
    # violated until the weight-update fault tolerance work removes the need
    async def stop_cells(self, *, cell_ids: list[str]) -> None:
        await self._client.stop_cells(cell_ids)

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return self._client.get_worker_infos_blocking(cell_ids)

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return await self._client.get_worker_addrs(worker_name)

    def get_handles_of_worker_infos(self, infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
        native_handles = self._client.get_native_handles_blocking(
            [info for info in infos if info.worker_class is None]
        )
        return {
            info.name: (
                native_handles[info.name] if info.worker_class is None else build_rpc_handle_of_worker_info(info)
            )
            for info in infos
        }

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        loop = PollingReconcileLoop(
            list_cells=partial(self._list_alive_cells, pool_ids=self._pool_ids),
            poll_interval_seconds=self._poll_interval_seconds,
        )
        return await loop.start(reconcile)

    async def _list_alive_cells(self, *, pool_ids: list[str] | None) -> dict[str, CellInfo]:
        all_infos = await self._client.get_cell_infos(pool_ids=pool_ids, category=self._category)
        return {cell_id: info for cell_id, info in all_infos.items() if info.alive}


class ManagerCellOperations(BaseCellOperations):
    def __init__(self, client: WorkerManagerClient) -> None:
        self._client = client

    async def cell_infos(self, *, pool_ids: list[str] | None, category: str | None) -> dict[str, CellInfo]:
        return await self._client.get_cell_infos(pool_ids=pool_ids, category=category)

    async def suspend(self, *, cell_id: str) -> None:
        await self._client.stop_cells([cell_id])

    async def resume(self, *, cell_id: str) -> None:
        await self._client.start_cells([cell_id])

    async def observe_fault_target(self, *, cell_id: str, rank: int) -> ObservedFaultHookTarget:
        return await self._client.observe_fault_target(cell_id, rank=rank)

    async def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord:
        return await asyncio.wait_for(
            self._client.control_fault_hook(command), timeout=CONTROL_FAULT_HOOK_TIMEOUT_SECONDS
        )


class ManagerBackendCapability(BackendCapability):
    def __init__(self, client: WorkerManagerClient) -> None:
        self._client = client

    def dynamic_worker_provider(
        self, *, pool_ids: Sequence[str] | None, category: str | None = None
    ) -> BaseWorkerProvider:
        return ManagerWorkerProvider(
            self._client, pool_ids=list(pool_ids) if pool_ids is not None else None, category=category
        )

    def static_worker_provider(self, *, pool_id: str) -> BaseWorkerProvider:
        return ManagerWorkerProvider(self._client)

    def cell_operations(self) -> BaseCellOperations:
        return ManagerCellOperations(self._client)
