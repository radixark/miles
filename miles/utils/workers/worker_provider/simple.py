from __future__ import annotations

from miles.utils.workers.naming import parse_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo
from miles.utils.workers.worker_provider.utils import WorkerClassLoader, build_rpc_handle
from miles.utils.workers.worker_spec import NamedHostAndPorts


class SimpleWorkerProvider(BaseWorkerProvider):
    def __init__(
        self,
        *,
        addrs: dict[str, NamedHostAndPorts],
        cells: dict[str, list[str]],
        pool_ids: dict[str, str],
        worker_classes: dict[str, str] | None = None,
    ) -> None:
        self._addrs = addrs
        self._cells = cells
        self._pools = pool_ids
        self._worker_classes = WorkerClassLoader(worker_classes or {})

    def knows_pool(self, pool_id: str) -> bool:
        return pool_id in set(self._pool_ids.values())

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        addrs = self._addrs.get(worker_name)
        assert addrs is not None, f"worker {worker_name} is not in the address book: {sorted(self._addrs)}"
        return addrs

    async def cell_infos(self, *, pool_id: str) -> dict[str, CellInfo]:
        infos = {cell_id: self.cell_info(cell_id) for cell_id in self.cell_ids()}
        return {cell_id: info for cell_id, info in infos.items() if info is not None and info.pool_id == pool_id}

    def cell_ids(self) -> list[str]:
        return sorted(self._cells)

    def cell_info(self, cell_id: str) -> CellInfo | None:
        worker_names = self._cells.get(cell_id)
        if worker_names is None:
            return None

        pool_id = self._pools.get(cell_id)
        assert pool_id is not None, f"cell {cell_id} is in the address book without a spec name"
        return CellInfo(
            cell_id=cell_id,
            pool_id=pool_id,
            alive=True,
            worker_names=list(worker_names),
            workers_hash=f"static-{cell_id}",
            meta={},
        )

    def _worker_infos_of_cell(self, cell_id: str) -> list[WorkerInfo]:
        worker_names = self._cells.get(cell_id)
        assert worker_names, f"cell {cell_id} is not in the address book: {sorted(self._cells)}"
        return [
            WorkerInfo(
                name=worker_name,
                generation=0,
                self_addrs=self._addrs[worker_name],
                gpu_ids=[],
                handle=self.get_handle(worker_name),
            )
            for worker_name in worker_names
        ]

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [self._worker_infos_of_cell(cell_id) for cell_id in cell_ids]

    def get_handle(self, worker_name: str) -> BaseWorkerHandle:
        addrs = self._addrs.get(worker_name)
        assert addrs is not None, f"worker {worker_name} is not in the address book: {sorted(self._addrs)}"
        pool_id = parse_worker_name(worker_name)[0]
        return build_rpc_handle(worker_class=self._worker_classes.of_spec(pool_id), addrs=addrs, pool_id=pool_id)
