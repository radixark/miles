from argparse import Namespace
from collections.abc import Sequence
from typing import Any, Literal

import torch

from miles.backends.training_utils.data import DataIterator
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import TrainerCpuWitnessEvent, TrainerGroupMappingEvent
from miles.utils.audit_utils.witness.cpu import cpu_witnesses
from miles.utils.ft_utils.process_group_utils import GeneralPGUtil


def collect_consumed_groups(iterator: DataIterator, *, start: int) -> list[list[int]]:
    data = iterator.rollout_data
    if "group_indices" not in data:
        raise ValueError("Training data is missing group_indices for the CPU witness")
    indices = (
        [index for batch in iterator.micro_batch_indices[start : iterator.offset] for index in batch]
        if iterator.micro_batch_indices is not None
        else list(range(start, iterator.offset))
    )
    slots = data.get("adapter_slots")
    rows = [
        [data["sample_indices"][index], data["group_indices"][index], slots[index] if slots is not None else -1]
        for index in indices
    ]
    parallel = get_parallel_state()
    if parallel.effective_dp.size == 1:
        return rows

    group = parallel.effective_dp.gloo_group
    assert group is not None, "CPU witness requires the effective DP Gloo group"
    util = GeneralPGUtil.create(group)
    count = torch.tensor([len(rows)], dtype=torch.int64)
    counts = [torch.empty_like(count) for _ in range(parallel.effective_dp.size)]
    util.all_gather(output_tensors=counts, input_tensor=count, group=group)
    padded = torch.full((max(int(value.item()) for value in counts), 3), -1, dtype=torch.int64)
    if rows:
        padded[: len(rows)] = torch.tensor(rows, dtype=torch.int64)
    gathered = [torch.empty_like(padded) for _ in counts]
    util.all_gather(output_tensors=gathered, input_tensor=padded, group=group)
    return [row for tensor, size in zip(gathered, counts, strict=True) for row in tensor[: int(size.item())].tolist()]


def record_optimizer_step(
    *,
    args: Namespace,
    model: Sequence[torch.nn.Module],
    rows: list[list[int]],
    rollout_data: dict[str, Any],
    rollout_id: int,
    step_id: int,
    attempt: int,
) -> None:
    lineage_id = rollout_data["ownership_lineage_id"]
    by_slot: dict[int, dict[int, list[int]]] = {}
    for sample_index, group_index, slot in rows:
        by_slot.setdefault(slot, {}).setdefault(group_index, []).append(sample_index)

    witnesses = cpu_witnesses(model)
    assert witnesses, "Model is missing its CPU training witness"
    for slot, groups in by_slot.items():
        fields = dict(
            lineage_id=lineage_id,
            trainer_model_id=args.trainer_model_id,
            rollout_id=rollout_id,
            step_id=step_id,
            attempt=attempt,
            slot=None if slot == -1 else slot,
        )
        for witness in witnesses:
            witness.commit(dict(**fields, group_indices=list(groups)), slot=fields["slot"])
        if is_event_logger_initialized():
            get_event_logger().log(TrainerGroupMappingEvent, dict(**fields, groups=groups), print_log=False)

    for witness in witnesses:
        witness.step_slots(list(rollout_data.get("step_adapter_batch_sizes", {})))
    log_cpu_witness(model=model, lineage_id=lineage_id, trainer_model_id=args.trainer_model_id, rollout_id=rollout_id)


def log_cpu_witness(
    *,
    model: Sequence[torch.nn.Module],
    lineage_id: str | None,
    trainer_model_id: str | None,
    rollout_id: int,
    reset: bool = False,
    reason: Literal["step", "train_end", "save", "transfer", "load"] = "step",
    checkpoint_id: str | None = None,
) -> None:
    if not is_event_logger_initialized():
        return
    witnesses = cpu_witnesses(model)
    if not witnesses:
        return
    witness = witnesses[0]
    reset = reset or witness.emitted_record_count is None
    records = witness.records if reset else witness.records[witness.emitted_record_count :]
    get_event_logger().log(
        TrainerCpuWitnessEvent,
        dict(
            lineage_id=lineage_id,
            trainer_model_id=trainer_model_id,
            rollout_id=rollout_id,
            records=records,
            pending_records=[record for records in witness.pending.values() for record in records],
            reset=reset,
            reason=reason,
            checkpoint_id=checkpoint_id,
        ),
        print_log=False,
    )
    witness.emitted_record_count = len(witness.records)
