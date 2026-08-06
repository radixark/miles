"""In-process stand-in for the RayWorkerManager actor handle used by trainer cells."""

import ray
from tests.fast.ray.train.dummy_actor import DummyTrainActor

from miles.ray.specs.train import MASTER_PORT_NAME
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.ray_worker_manager import WorkerInfo
from miles.utils.workers.worker_spec import HostAndPort


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

        self.get_cell_infos = _FakeRemoteMethod(self._get_cell_infos)
        self.get_worker_infos = _FakeRemoteMethod(self._get_worker_infos)
        self.start_cells = _FakeRemoteMethod(self.started_cell_ids.append)
        self.stop_cells = _FakeRemoteMethod(self._stop_cells)

    def fail_init_for_cell(self, cell_index: int) -> None:
        self._cell_indices_failing_init.add(cell_index)

    def _get_cell_infos(self, *, spec_names: list[str]) -> dict:
        return {compute_cell_id(spec_name=spec_names[0], cell_index=index): None for index in range(self.num_cells)}

    def _get_worker_infos(self, spec_name: str, cell_index: int) -> list[WorkerInfo]:
        cell_id = compute_cell_id(spec_name=spec_name, cell_index=cell_index)
        if cell_id not in self._handles:
            handles = [DummyTrainActor.remote() for _ in range(self.actor_count_per_cell)]
            if cell_index in self._cell_indices_failing_init:
                ray.get([handle.set_fail_methods.remote(["init"]) for handle in handles])
            self._handles[cell_id] = handles
        return [
            WorkerInfo(
                name=f"{cell_id}-{worker_index}",
                generation=1 + len(self.started_cell_ids),
                self_addrs={MASTER_PORT_NAME: HostAndPort(host="10.0.0.1", port=20000)},
                gpu_ids=[worker_index],
                actor_handle=handle,
            )
            for worker_index, handle in enumerate(self._handles[cell_id])
        ]

    def _stop_cells(self, cell_ids: list[str]) -> None:
        self.stopped_cell_ids.append(cell_ids)
        for cell_id in cell_ids:
            for handle in self._handles.pop(cell_id, []):
                ray.kill(handle)
