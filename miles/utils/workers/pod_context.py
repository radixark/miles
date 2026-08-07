from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.worker_spec import WorkerCtorContext

SUBPROCESS_INDEX_ENV_VAR = "MILES_SUPERVISOR_SUBPROCESS_INDEX"
LEADER_ADDRESS_ENV_VAR = "LWS_LEADER_ADDRESS"
POD_INDEX_ENV_VAR = "LWS_WORKER_INDEX"


@dataclass(frozen=True)
class PodRank:
    cell_ordinal: int
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

    def ctor_context(self, *, pool_id: str, capability: BackendCapability) -> WorkerCtorContext:
        return WorkerCtorContext(
            cell_id=compute_cell_id(pool_id=pool_id, cell_index=self.cell_ordinal),
            cell_ordinal=self.cell_ordinal,
            worker_in_cell_index=self.worker_in_cell_index,
            gpu_ids=self.gpu_ids,
            capability=capability,
        )


def read_pod_rank(*, ranks_per_pod: int, gpu_slots_per_rank: int, environ: Mapping[str, str] | None = None) -> PodRank:
    environ = os.environ if environ is None else environ
    rank_in_pod = _index_from(environ, SUBPROCESS_INDEX_ENV_VAR)
    assert rank_in_pod < ranks_per_pod, (
        f"{SUBPROCESS_INDEX_ENV_VAR} is {rank_in_pod} in a pod launched for {ranks_per_pod} ranks; the rank this "
        f"process reports would collide with another pod's"
    )
    return PodRank(
        cell_ordinal=read_cell_ordinal(environ),
        pod_index=_index_from(environ, POD_INDEX_ENV_VAR),
        rank_in_pod=rank_in_pod,
        ranks_per_pod=ranks_per_pod,
        gpu_slots_per_rank=gpu_slots_per_rank,
    )


def read_cell_ordinal(environ: Mapping[str, str]) -> int:
    leader_address = environ.get(LEADER_ADDRESS_ENV_VAR, "")
    if not leader_address:
        return 0

    leader_host = leader_address.split(".", maxsplit=1)[0]
    _, separator, cell_ordinal = leader_host.rpartition("-")
    assert separator and cell_ordinal.isdigit(), (
        f"{LEADER_ADDRESS_ENV_VAR} is {leader_address!r}, which does not end in the ordinal the pool_id "
        f"leader's hostname carries, so this pod cannot tell which cell it belongs to"
    )
    return int(cell_ordinal)


def _index_from(environ: Mapping[str, str], name: str) -> int:
    value = environ.get(name, "0")
    assert value.isdigit(), f"{name} is {value!r}, which is not an index"
    return int(value)
