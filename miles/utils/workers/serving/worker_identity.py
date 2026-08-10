from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.process_supervisor import SUBPROCESS_INDEX_ENV_VAR
from miles.utils.workers.worker_spec import SchedulingSpec, WorkerCtorContext

CELL_INDEX_ENV_VAR = "MILES_CELL_INDEX"
POD_INDEX_ENV_VAR = "MILES_POD_INDEX"


@dataclass(frozen=True)
class KubernetesWorkerIdentity:
    cell_index: int
    pod_in_cell_index: int
    worker_in_pod_index: int
    workers_per_pod: int
    gpu_slots_per_worker: int

    @property
    def worker_in_cell_index(self) -> int:
        return self.pod_in_cell_index * self.workers_per_pod + self.worker_in_pod_index

    @property
    def gpu_ids(self) -> list[int]:
        first = self.worker_in_pod_index * self.gpu_slots_per_worker
        return list(range(first, first + self.gpu_slots_per_worker))

    def ctor_context(self, *, capability: BackendCapability) -> WorkerCtorContext:
        return WorkerCtorContext(
            cell_index=self.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
            gpu_ids=self.gpu_ids,
            capability=capability,
        )


def read_worker_identity(
    *, scheduling: SchedulingSpec, environ: Mapping[str, str] | None = None
) -> KubernetesWorkerIdentity:
    environ = os.environ if environ is None else environ

    workers_per_pod = scheduling.workers_per_pod()
    pods_per_cell = scheduling.pods_per_cell()

    worker_in_pod_index = read_worker_in_pod_index(environ)
    assert worker_in_pod_index < workers_per_pod, (
        f"{SUBPROCESS_INDEX_ENV_VAR} is {worker_in_pod_index} in a pod launched for {workers_per_pod} workers; "
        f"the worker this process reports would collide with another pod's"
    )

    pod_in_cell_index = _index_from(
        environ,
        POD_INDEX_ENV_VAR,
        required_because=(
            f"this pod's cell is spread over {pods_per_cell} pods, and a pod that read zero here would "
            f"claim the workers of the pod that leads the cell"
            if pods_per_cell > 1
            else None
        ),
    )
    assert pod_in_cell_index < pods_per_cell, (
        f"{POD_INDEX_ENV_VAR} is {pod_in_cell_index} in a cell spread over {pods_per_cell} pods; "
        f"the workers this pod reports would belong to no cell of this pool"
    )

    return KubernetesWorkerIdentity(
        cell_index=_index_from(
            environ,
            CELL_INDEX_ENV_VAR,
            required_because="nothing else tells this pod which cell of its pool it belongs to",
        ),
        pod_in_cell_index=pod_in_cell_index,
        worker_in_pod_index=worker_in_pod_index,
        workers_per_pod=workers_per_pod,
        gpu_slots_per_worker=scheduling.num_gpu_slots_per_worker,
    )


def read_worker_in_pod_index(environ: Mapping[str, str] | None = None) -> int:
    environ = os.environ if environ is None else environ
    return _index_from(environ, SUBPROCESS_INDEX_ENV_VAR, required_because=None)


def _index_from(environ: Mapping[str, str], name: str, *, required_because: str | None) -> int:
    value = environ.get(name)
    assert value is not None or required_because is None, f"{name} is not set, but {required_because}"

    value = "0" if value is None else value
    assert value.isdigit(), f"{name} is {value!r}, which is not an index"
    return int(value)
