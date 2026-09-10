import copy
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch


class CpuWitness(torch.nn.Module):
    def __init__(self, *, pipeline_rank: int, chunk_index: int, replica_id: tuple[int, ...]) -> None:
        super().__init__()
        self.pipeline_rank = pipeline_rank
        self.chunk_index = chunk_index
        self.replica_id = replica_id
        self.records: list[dict[str, Any]] = []
        self.pending: dict[int, list[dict[str, Any]]] = {}
        self.emitted_record_count: int | None = None

    def commit(self, record: dict[str, Any], *, slot: int | None = None) -> None:
        record = dict(record, adapter_slot=slot)
        if slot is None:
            self.records.append(copy.deepcopy(record))
        else:
            self.pending.setdefault(slot, []).append(copy.deepcopy(record))

    def step_slots(self, slots: Sequence[int]) -> None:
        for slot in slots:
            self.records.extend(self.pending.pop(slot, []))

    def get_extra_state(self) -> dict[str, Any]:
        return copy.deepcopy(dict(version=1, records=self.records, pending=self.pending))

    def set_extra_state(self, state: dict[str, Any]) -> None:
        assert state["version"] == 1, f"Unsupported CPU witness version: {state['version']}"
        self.records = copy.deepcopy(state["records"])
        self.pending = copy.deepcopy(state["pending"])
        self.emitted_record_count = None

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


def install_cpu_witness(model: torch.nn.Module, *, chunk_index: int | None) -> None:
    from miles.backends.training_utils.parallel import get_parallel_state

    parallel = get_parallel_state()
    model.add_module(
        "cpu_witness",
        CpuWitness(
            pipeline_rank=parallel.pp.rank,
            chunk_index=chunk_index or 0,
            replica_id=(parallel.tp.rank, parallel.cp.rank, parallel.effective_dp.rank),
        ),
    )


def cpu_witnesses(model: Sequence[torch.nn.Module]) -> list[CpuWitness]:
    return [module for chunk in model for module in chunk.modules() if isinstance(module, CpuWitness)]


def snapshot_cpu_witness(model: Sequence[torch.nn.Module]) -> list[dict[str, Any]]:
    return [witness.get_extra_state() for witness in cpu_witnesses(model)]


def restore_cpu_witness(model: Sequence[torch.nn.Module], states: list[dict[str, Any]]) -> None:
    witnesses = cpu_witnesses(model)
    assert len(witnesses) == len(states), "CPU witness model chunk count changed"
    for witness, state in zip(witnesses, states, strict=True):
        witness.set_extra_state(state)


def clear_cpu_witness(model: Sequence[torch.nn.Module]) -> None:
    for witness in cpu_witnesses(model):
        witness.set_extra_state(dict(version=1, records=[], pending={}))


def snapshot_adapter_cpu_witness(model: Sequence[torch.nn.Module], *, slot: int) -> list[dict[str, Any]]:
    states = snapshot_cpu_witness(model)
    for state in states:
        assert not state["pending"].get(slot), "Cannot checkpoint an adapter with uncommitted gradients"
        state["records"] = [record for record in state["records"] if record["adapter_slot"] == slot]
        state["pending"] = {}
    return states


def restore_adapter_cpu_witness(model: Sequence[torch.nn.Module], *, states: list[dict[str, Any]], slot: int) -> None:
    witnesses = cpu_witnesses(model)
    assert len(witnesses) == len(states), "CPU witness model chunk count changed"
    for witness, state in zip(witnesses, states, strict=True):
        assert not state["pending"], "Adapter checkpoint contains gradients without gradient tensors"
        witness.pending.pop(slot, None)
        witness.records = [record for record in witness.records if record["adapter_slot"] != slot]
        witness.records.extend(dict(copy.deepcopy(record), adapter_slot=slot) for record in state["records"])
        witness.emitted_record_count = None


def clear_adapter_cpu_witness(model: Sequence[torch.nn.Module], *, slot: int) -> None:
    for witness in cpu_witnesses(model):
        witness.pending.pop(slot, None)
        witness.records = [record for record in witness.records if record["adapter_slot"] != slot]
        witness.emitted_record_count = None


@contextmanager
def preserve_cpu_witness(model: Sequence[torch.nn.Module]) -> Iterator[None]:
    state = snapshot_cpu_witness(model)
    try:
        yield
    finally:
        restore_cpu_witness(model=model, states=state)


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
