from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.process_supervisor import SUBPROCESS_INDEX_ENV_VAR
from miles.utils.workers.worker_spec import SchedulingSpec, WorkerCtorContext

CELL_INDEX_ENV_VAR = "MILES_CELL_INDEX"
POD_INDEX_ENV_VAR = "LWS_WORKER_INDEX"


@dataclass(frozen=True)
class PodRank:
    cell_index: int
    pod_index: int
    rank_in_pod: int
    ranks_per_pod: int
    gpu_slots_per_rank: int

    @property
    def worker_in_cell_index(self) -> int:
        return self.pod_index * self.ranks_per_pod + self.rank_in_pod

    @property
    def gpu_ids(self) -> list[int]:
        first = self.rank_in_pod * self.gpu_slots_per_rank
        return list(range(first, first + self.gpu_slots_per_rank))

    def ctor_context(self, *, capability: BackendCapability) -> WorkerCtorContext:
        return WorkerCtorContext(
            cell_index=self.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
            gpu_ids=self.gpu_ids,
            capability=capability,
        )


def read_pod_rank(*, scheduling: SchedulingSpec, environ: Mapping[str, str] | None = None) -> PodRank:
    environ = os.environ if environ is None else environ

    ranks_per_pod = scheduling.ranks_per_pod()
    pods_per_cell = scheduling.pods_per_cell()

    rank_in_pod = read_rank_in_pod(environ)
    assert rank_in_pod < ranks_per_pod, (
        f"{SUBPROCESS_INDEX_ENV_VAR} is {rank_in_pod} in a pod launched for {ranks_per_pod} ranks; the rank "
        f"this process reports would collide with another pod's"
    )

    return PodRank(
        cell_index=_index_from(
            environ,
            CELL_INDEX_ENV_VAR,
            required_because="nothing else tells this pod which cell of its pool it belongs to",
        ),
        pod_index=_index_from(
            environ,
            POD_INDEX_ENV_VAR,
            required_because=(
                f"this pod's cell is spread over {pods_per_cell} pods, and a pod that read zero here would "
                f"claim the ranks of the pod that leads the cell"
                if pods_per_cell > 1
                else None
            ),
        ),
        rank_in_pod=rank_in_pod,
        ranks_per_pod=ranks_per_pod,
        gpu_slots_per_rank=scheduling.num_gpu_slots_per_worker,
    )


def read_rank_in_pod(environ: Mapping[str, str] | None = None) -> int:
    environ = os.environ if environ is None else environ
    return _index_from(environ, SUBPROCESS_INDEX_ENV_VAR, required_because=None)


def _index_from(environ: Mapping[str, str], name: str, *, required_because: str | None) -> int:
    value = environ.get(name)
    assert value is not None or required_because is None, f"{name} is not set, but {required_because}"

    value = "0" if value is None else value
    assert value.isdigit(), f"{name} is {value!r}, which is not an index"
    return int(value)
