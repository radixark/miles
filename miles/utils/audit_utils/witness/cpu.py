import copy
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
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
        self.skipped_nonfinite_sample_counts: Counter[TrainingSampleIdentity] = Counter()

    def record(self, samples: Iterable[TrainingSampleIdentity]) -> None:
        self.sample_counts.update(samples)

    def snapshot(self) -> dict[TrainingSampleIdentity, int]:
        return dict(self.sample_counts)

    def record_skipped_nonfinite(self, samples: Iterable[TrainingSampleIdentity]) -> None:
        self.skipped_nonfinite_sample_counts.update(samples)

    def snapshot_skipped_nonfinite(self) -> dict[TrainingSampleIdentity, int]:
        return dict(self.skipped_nonfinite_sample_counts)

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 2,
            "sample_counts": self.snapshot(),
            "skipped_nonfinite_sample_counts": self.snapshot_skipped_nonfinite(),
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        assert state["version"] in (1, 2), f"Unsupported CPU witness version: {state['version']}"
        self.sample_counts = Counter(copy.deepcopy(state["sample_counts"]))
        skipped_nonfinite_sample_counts = {} if state["version"] == 1 else state["skipped_nonfinite_sample_counts"]
        self.skipped_nonfinite_sample_counts = Counter(copy.deepcopy(skipped_nonfinite_sample_counts))

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
            replica_id=(parallel.tp.rank, parallel.cp.rank, parallel.intra_dp.rank),
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


def record_nonfinite_skip_cpu_witness(
    model: Sequence[torch.nn.Module], samples: Iterable[TrainingSampleIdentity]
) -> None:
    identities = list(samples)
    for witness in cpu_witnesses(model):
        witness.record_skipped_nonfinite(identities)


def snapshot_nonfinite_skip_cpu_witness(
    model: Sequence[torch.nn.Module],
) -> dict[TrainingSampleIdentity, int]:
    witnesses = cpu_witnesses(model)
    assert witnesses, "Model is missing its CPU training witness"
    snapshots = [witness.snapshot_skipped_nonfinite() for witness in witnesses]
    assert all(snapshot == snapshots[0] for snapshot in snapshots[1:]), "CPU witness model chunks diverged"
    return snapshots[0]


def clear_cpu_witness(model: Sequence[torch.nn.Module]) -> None:
    for witness in cpu_witnesses(model):
        witness.set_extra_state({"version": 2, "sample_counts": {}, "skipped_nonfinite_sample_counts": {}})


@contextmanager
def preserve_cpu_witness(model: Sequence[torch.nn.Module]) -> Iterator[None]:
    states = [witness.get_extra_state() for witness in cpu_witnesses(model)]
    try:
        yield
    finally:
        for witness, state in zip(cpu_witnesses(model), states, strict=True):
            witness.set_extra_state(state)


@contextmanager
def hide_cpu_witness(model: Sequence[torch.nn.Module]) -> Iterator[None]:
    children = [
        (parent, name, child)
        for chunk in model
        for parent in list(chunk.modules())
        for name, child in list(parent.named_children())
        if isinstance(child, CpuWitness)
    ]
    for parent, name, _ in children:
        delattr(parent, name)
    try:
        yield
    finally:
        for parent, name, child in children:
            parent.add_module(name, child)
