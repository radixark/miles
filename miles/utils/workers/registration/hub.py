from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from miles.utils.misc import cancel_and_await_task
from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.naming import cell_id_of_worker, parse_cell_id
from miles.utils.workers.polling_reconcile_loop import PollingReconcileLoop
from miles.utils.workers.registration.models import RegisteredCellInfo, RegistrationSnapshot
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import NamedHostAndPorts

logger = logging.getLogger(__name__)

REGISTERED_CELLS_POLL_INTERVAL_SECONDS = 5.0
REPORTER_TTL_SECONDS = 240.0


@dataclass(kw_only=True)
class RegistrationHub(BaseWorkerProvider):
    run_uuid: str
    clock: Callable[[], float] = time.monotonic
    _state_of_reporter_id: dict[str, _ReporterState] = field(init=False, default_factory=dict)
    _cell_of_id: dict[str, RegisteredCellInfo] = field(init=False, default_factory=dict)
    _watched: bool = field(init=False, default=False)

    # ========================== Taking in snapshots ===========================

    async def ingest(self, snapshot: RegistrationSnapshot) -> None:
        assert (
            snapshot.run_uuid == self.run_uuid
        ), f"reporter {snapshot.reporter_id} carries run_uuid {snapshot.run_uuid!r}, expected {self.run_uuid!r}"
        _assert_snapshot_addressable(snapshot)
        cell_of_id = {cell.info.cell_id: cell for cell in snapshot.cells}

        state = self._state_of_reporter_id.setdefault(snapshot.reporter_id, _ReporterState())

        if snapshot.sequence_number <= state.sequence_number:
            logger.warning(
                f"Ignoring snapshot {snapshot.sequence_number} of reporter {snapshot.reporter_id}: snapshot "
                f"{state.sequence_number} is at least as new, so this one arrived late"
            )
            return

        state.last_ingest_time = self.clock()
        self._replace_cells_of_reporter(reporter_id=snapshot.reporter_id, cell_of_id=cell_of_id)
        state.sequence_number = snapshot.sequence_number

    def _replace_cells_of_reporter(self, *, reporter_id: str, cell_of_id: dict[str, RegisteredCellInfo]) -> None:
        for cell_id in cell_of_id:
            assert (
                owner := self._cell_of_id.get(cell_id)
            ) is None or owner.reporter_id == reporter_id, (
                f"cell {cell_id} is reported by both {owner.reporter_id} and {reporter_id}"
            )

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
        async with contextlib.AsyncExitStack() as stack:
            stack.push_async_callback(await loop.start(reconcile))
            sweep_task = asyncio.create_task(self._remove_stale_reporters_forever())
            stack.push_async_callback(partial(cancel_and_await_task, sweep_task))
            return stack.pop_all().aclose

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

    async def _remove_stale_reporters_forever(self) -> None:
        while True:
            await asyncio.sleep(REGISTERED_CELLS_POLL_INTERVAL_SECONDS)
            self._remove_stale_reporters()

    def _remove_stale_reporters(self) -> None:
        deadline = self.clock() - REPORTER_TTL_SECONDS
        for reporter_id, state in list(self._state_of_reporter_id.items()):
            if state.last_ingest_time <= deadline:
                self._remove_reporter(reporter_id)

    def _remove_reporter(self, reporter_id: str) -> None:
        logger.warning(
            f"Dropping reporter {reporter_id} and every cell it announced: it has not reported for "
            f"{REPORTER_TTL_SECONDS}s, and this run serves requests to engines only while the deployment that "
            f"carries them keeps saying they are there"
        )
        del self._state_of_reporter_id[reporter_id]
        for cell_id in [cell_id for cell_id, cell in self._cell_of_id.items() if cell.reporter_id == reporter_id]:
            del self._cell_of_id[cell_id]


class _ReporterState(StrictBaseModel):
    sequence_number: int = -1
    last_ingest_time: float = 0.0


def _assert_snapshot_addressable(snapshot: RegistrationSnapshot) -> None:
    occurrences = Counter(cell.info.cell_id for cell in snapshot.cells)
    repeated = sorted(cell_id for cell_id, count in occurrences.items() if count > 1)
    assert not repeated, f"reporter {snapshot.reporter_id} carries the cells {repeated} more than once"

    for cell in snapshot.cells:
        _assert_cell_addressable(cell, reporter_id=snapshot.reporter_id)


def _assert_cell_addressable(cell: RegisteredCellInfo, *, reporter_id: str) -> None:
    assert (
        cell.reporter_id == reporter_id
    ), f"snapshot of reporter {reporter_id} carries cell {cell.info.cell_id} of reporter {cell.reporter_id}"

    prefix = f"cell {cell.info.cell_id} of reporter {cell.reporter_id} is not addressable:"
    try:
        pool_id = parse_cell_id(cell.info.cell_id).pool_id
        worker_cell_ids = {cell_id_of_worker(worker.name) for worker in cell.workers}
    except ValueError as cause:
        raise AssertionError(
            f"{prefix} its cell id or a worker name does not read as <pool id>-<cell index>"
        ) from cause

    assert pool_id == cell.info.pool_id, f"{prefix} its cell id names pool {pool_id}, not {cell.info.pool_id}"
    assert cell.workers, f"{prefix} it carries no worker"
    assert worker_cell_ids == {
        cell.info.cell_id
    }, f"{prefix} its workers {sorted(one.name for one in cell.workers)} name cells {sorted(worker_cell_ids)}"
