from __future__ import annotations

import logging
from dataclasses import dataclass, field

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.naming import cell_id_of_worker
from miles.utils.workers.polling_reconcile_loop import PollingReconcileLoop
from miles.utils.workers.registration.models import RegisteredCellInfo, RegistrationSnapshot
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import NamedHostAndPorts

logger = logging.getLogger(__name__)

REGISTERED_CELLS_POLL_INTERVAL_SECONDS = 5.0


@dataclass(kw_only=True)
class RegistrationHub(BaseWorkerProvider):
    _state_of_reporter_id: dict[str, _ReporterState] = field(init=False, default_factory=dict)
    _cell_of_id: dict[str, RegisteredCellInfo] = field(init=False, default_factory=dict)
    _watched: bool = field(init=False, default=False)

    # ========================== Taking in snapshots ===========================

    async def ingest(self, snapshot: RegistrationSnapshot) -> None:
        cell_of_id = {cell.info.cell_id: cell for cell in snapshot.cells}

        state = self._state_of_reporter_id.setdefault(snapshot.reporter_id, _ReporterState())
        if snapshot.sequence_number <= state.sequence_number:
            logger.warning(
                f"Ignoring snapshot {snapshot.sequence_number} of reporter {snapshot.reporter_id}: snapshot "
                f"{state.sequence_number} is at least as new, so this one arrived late"
            )
            return

        self._replace_cells_of_reporter(reporter_id=snapshot.reporter_id, cell_of_id=cell_of_id)
        state.sequence_number = snapshot.sequence_number

    def _replace_cells_of_reporter(self, *, reporter_id: str, cell_of_id: dict[str, RegisteredCellInfo]) -> None:
        held = {cell_id for cell_id, cell in self._cell_of_id.items() if cell.reporter_id == reporter_id}
        for cell_id in held - set(cell_of_id):
            del self._cell_of_id[cell_id]
        self._cell_of_id.update(cell_of_id)

    # ======================= Serving the reported cells =======================

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        assert not self._watched, "a registration hub reports to exactly one watcher"
        self._watched = True

        async def _list_cells() -> dict[str, CellInfo]:
            return {cell_id: cell.info for cell_id, cell in self._cell_of_id.items()}

        loop = PollingReconcileLoop(
            list_cells=_list_cells,
            poll_interval_seconds=REGISTERED_CELLS_POLL_INTERVAL_SECONDS,
        )
        return await loop.start(reconcile)

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        cell = self._cell_of_id[cell_id_of_worker(worker_name)]
        worker = next((one for one in cell.workers if one.name == worker_name), None)
        assert worker is not None, (
            f"{worker_name} is not among the workers {sorted(one.name for one in cell.workers)} "
            f"reporter {cell.reporter_id} announced"
        )
        return worker.self_addrs

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [self._cell_of_id[cell_id].workers for cell_id in cell_ids]


class _ReporterState(StrictBaseModel):
    sequence_number: int = -1
