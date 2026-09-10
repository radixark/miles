import copy
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TrainingSampleIdentity:
    source_sample_index: int
    row_index: int
    row_count: int


class CpuWitness(torch.nn.Module):
    def __init__(self, *, pipeline_rank: int = 0, chunk_index: int = 0, replica_id: tuple[int, ...] = (0,)) -> None:
        super().__init__()
        self.pipeline_rank = pipeline_rank
        self.chunk_index = chunk_index
        self.replica_id = replica_id
        self.sample_counts: Counter[TrainingSampleIdentity] = Counter()

    def record(self, samples: Iterable[TrainingSampleIdentity]) -> None:
        self.sample_counts.update(samples)

    def snapshot(self) -> dict[TrainingSampleIdentity, int]:
        return dict(self.sample_counts)

    def get_extra_state(self) -> dict[str, Any]:
        return {"version": 1, "sample_counts": self.snapshot()}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        assert state["version"] == 1, f"Unsupported CPU witness version: {state['version']}"
        self.sample_counts = Counter(copy.deepcopy(state["sample_counts"]))

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> dict[str, Any]:
        from megatron.core.dist_checkpointing.mapping import ShardedObject

        return {
            f"{prefix}_extra_state": ShardedObject(
                key=f"{prefix}pp{self.pipeline_rank}.chunk{self.chunk_index}._extra_state",
                data=self.get_extra_state(),
                global_shape=(1,),
                global_offset=(0,),
                replica_id=self.replica_id,
            )
        }


def install_cpu_witness(model: torch.nn.Module, *, chunk_index: int) -> None:
    from miles.backends.training_utils.parallel import get_parallel_state

    parallel = get_parallel_state()
    model.add_module(
        "cpu_witness",
        CpuWitness(
            pipeline_rank=parallel.pp.rank,
            chunk_index=chunk_index,
            replica_id=(parallel.tp.rank, parallel.cp.rank, parallel.effective_dp.rank),
        ),
    )


def cpu_witnesses(model: Sequence[torch.nn.Module]) -> list[CpuWitness]:
    return [module for chunk in model for module in chunk.modules() if isinstance(module, CpuWitness)]


def record_cpu_witness(model: Sequence[torch.nn.Module], samples: Iterable[TrainingSampleIdentity]) -> None:
    identities = list(samples)
    for witness in cpu_witnesses(model):
        witness.record(identities)


def snapshot_cpu_witness(model: Sequence[torch.nn.Module]) -> dict[TrainingSampleIdentity, int]:
    witnesses = cpu_witnesses(model)
    assert witnesses, "Model is missing its CPU training witness"
    snapshots = [witness.snapshot() for witness in witnesses]
    assert all(snapshot == snapshots[0] for snapshot in snapshots[1:]), "CPU witness model chunks diverged"
    return snapshots[0]
