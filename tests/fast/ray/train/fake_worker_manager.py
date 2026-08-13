"""In-process stand-in for the RayWorkerManager actor handle used by trainer cells."""

import logging

import ray
from tests.fast.ray.train.dummy_actor import DummyTrainActor

from miles.utils.workers.naming import compute_cell_id, parse_cell_id
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import MASTER_PORT_NAME, HostAndPort

logger = logging.getLogger(__name__)


class _FakeRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return ray.put(self._fn(*args, **kwargs))


class FakeWorkerManager:
    def __init__(self, *, num_cells: int = 1, actor_count_per_cell: int = 1):
        self.num_cells = num_cells
        self.actor_count_per_cell = actor_count_per_cell
        self.started_cell_ids: list[list[str]] = []
        self.stopped_cell_ids: list[list[str]] = []
        self._handles: dict[str, list] = {}
        self._cell_indices_failing_init: set[int] = set()
        self.master_addr_per_worker: list[HostAndPort] | None = None

        self.get_cell_infos = _FakeRemoteMethod(self._get_cell_infos)
        self.get_worker_infos = _FakeRemoteMethod(self._get_worker_infos)
        self.get_actor_handle = _FakeRemoteMethod(self._get_actor_handle)
        self.start_cells = _FakeRemoteMethod(self.started_cell_ids.append)
        self.stop_cells = _FakeRemoteMethod(self._stop_cells)

    def fail_init_for_cell(self, cell_index: int) -> None:
        self._cell_indices_failing_init.add(cell_index)

    def _get_cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        infos: dict[str, CellInfo] = {}
        for pool_id in pool_ids:
            for cell_index in range(self.num_cells):
                cell_id = compute_cell_id(pool_id=pool_id, cell_index=cell_index)
                infos[cell_id] = CellInfo(
                    cell_id=cell_id,
                    pool_id=pool_id,
                    alive=True,
                    worker_names=[f"{cell_id}-{worker_index}" for worker_index in range(self.actor_count_per_cell)],
                    workers_hash=f"pseudo-hash-{1 + len(self.started_cell_ids)}",
                    meta={"cell_index": cell_index},
                )
        return infos

    def _get_worker_infos(self, cell_id: str) -> list[WorkerInfo]:
        if cell_id not in self._handles:
            handles = [DummyTrainActor.remote() for _ in range(self.actor_count_per_cell)]
            if parse_cell_id(cell_id).cell_index in self._cell_indices_failing_init:
                ray.get([handle.set_fail_methods.remote(["init"]) for handle in handles])
            self._handles[cell_id] = handles
        return [
            WorkerInfo(
                name=f"{cell_id}-{worker_index}",
                generation=1 + len(self.started_cell_ids),
                self_addrs={MASTER_PORT_NAME: self._compute_master_addr(worker_index)},
                gpu_ids=[worker_index],
            )
            for worker_index, handle in enumerate(self._handles[cell_id])
        ]

    def _get_actor_handle(self, worker_name: str, *, expected_generation: int):
        generation = 1 + len(self.started_cell_ids)
        assert (
            generation == expected_generation
        ), f"{worker_name} is generation {generation}, not {expected_generation}"
        cell_id, _, worker_index = worker_name.rpartition("-")
        return self._handles[cell_id][int(worker_index)]

    def _compute_master_addr(self, worker_index: int) -> HostAndPort:
        if self.master_addr_per_worker is None:
            return HostAndPort(host="10.0.0.1", port=20000)
        return self.master_addr_per_worker[worker_index]

    def _stop_cells(self, cell_ids: list[str]) -> None:
        self.stopped_cell_ids.append(cell_ids)
        for cell_id in cell_ids:
            self._kill(self._handles.pop(cell_id, []))

    def kill_all_actors(self) -> None:
        for handles in self._handles.values():
            self._kill(handles)
        self._handles.clear()

    @staticmethod
    def _kill(handles: list) -> None:
        # The real manager kills the actor when it stops a cell. Dropping the handle
        # instead leaves the actor alive for as long as a cell still references it,
        # which accumulates one process per cell across a module.
        for handle in handles:
            try:
                ray.kill(handle)
            except Exception:
                logger.warning(f"Failed to kill {handle}", exc_info=True)
