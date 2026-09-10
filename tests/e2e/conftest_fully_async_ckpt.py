from pathlib import Path

import torch

from miles.ray.rollout.rollout_executor import compute_checkpoint_complete_marker_path, compute_executor_state_path
from miles.rollout.fully_async_data_buffer import iter_samples
from miles.rollout.fully_async_rollout import compute_fully_async_state_path
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership import current_lineage_id
from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import RolloutStateRestoreEvent, TrainerTrainedSamplesEvent


def read_checkpoint_sample_indices(*, save_dir: Path, rollout_id: int) -> set[int]:
    assert compute_checkpoint_complete_marker_path(save_dir, rollout_id=rollout_id).is_file()
    state = torch.load(compute_fully_async_state_path(save_dir, rollout_id=rollout_id), weights_only=False)
    groups = [pending.samples for pending in [*state["retry_buffer"], *state["in_flight_prompt_groups"]]]
    groups.extend(entry.group for entry in state["pending_puts"].values())
    groups.extend(entry.group for entries in state["output_buffer"].values() for entry in entries)
    executor_state = torch.load(compute_executor_state_path(save_dir, rollout_id=rollout_id), weights_only=False)
    groups.extend(group for batch in executor_state["last_batches"].values() for group in batch.samples)
    indices = {sample.index for group in groups for sample in iter_samples(group)}
    assert indices, f"Checkpoint {rollout_id} under {save_dir} has no owned samples to replay"
    return indices


def assert_checkpoint_replayed(
    *,
    event_dir: Path,
    saved_indices: set[int],
    rollout_ids: dict[str | None, int],
    num_rollout: int,
    leader_model_id: str | None = None,
) -> None:
    events = read_events(event_dir)
    lineage_id = current_lineage_id(events)
    restore = next(
        (event for event in events if isinstance(event, RolloutStateRestoreEvent) and event.lineage_id == lineage_id),
        None,
    )
    assert restore is not None, f"No restore event under {event_dir}"
    assert restore.rollout_id == rollout_ids[leader_model_id]
    if leader_model_id is not None:
        assert restore.rollout_ids == rollout_ids
    trained = [
        event for event in events if isinstance(event, TrainerTrainedSamplesEvent) and event.lineage_id == lineage_id
    ]
    assert saved_indices & {
        index for event in trained for index in event.sample_indices
    }, "No saved sample was replayed"
    for model_id, rollout_id in rollout_ids.items():
        policy_steps = {event.rollout_id for event in trained if event.trainer_model_id == model_id}
        assert rollout_id + 1 in policy_steps, f"Policy {model_id} did not resume after rollout {rollout_id}"
    assert any(
        event.rollout_id == num_rollout - 1 and event.trainer_model_id == leader_model_id for event in trained
    ), "The resumed run did not reach its final rollout"
