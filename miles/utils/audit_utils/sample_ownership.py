import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized, read_events
from miles.utils.audit_utils.event_logger.models import (
    RolloutGroupRoutedEvent,
    RolloutHoldingsSnapshotEvent,
    RolloutStateRestoreEvent,
    SampleOwner,
    SampleOwnerTransitionEvent,
    TrainerCheckpointEvent,
    TrainerTrainedSamplesEvent,
)
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity
from miles.utils.types import Sample

logger = logging.getLogger(__name__)
_lineage_id: str | None = None


def set_lineage_id(lineage_id: str) -> None:
    global _lineage_id
    _lineage_id = lineage_id


def log_owner_transition(
    prompt_group: Iterable[Sample],
    *,
    from_owner: SampleOwner,
    to_owner: SampleOwner,
    trainer_model_id: str | None = None,
    rollout_id: int | None = None,
    reason: str | None = None,
) -> None:
    if not is_event_logger_initialized():
        return

    samples = list(prompt_group)
    if not samples:
        return

    get_event_logger().log(
        SampleOwnerTransitionEvent,
        dict(
            lineage_id=_lineage_id,
            sample_indices=[sample.index for sample in samples if sample.index is not None],
            trainer_model_id=trainer_model_id,
            from_owner=from_owner,
            to_owner=to_owner,
            rollout_id=rollout_id,
            reason=reason,
        ),
        print_log=False,
    )


def log_group_routed(prompt_group: Iterable[Sample]) -> None:
    if not is_event_logger_initialized():
        return
    get_event_logger().log(
        RolloutGroupRoutedEvent,
        dict(
            lineage_id=_lineage_id,
            prompt_indices=[sample.index for sample in prompt_group if sample.index is not None],
        ),
        print_log=False,
    )


def log_holdings_snapshot(
    *,
    rollout_id: int,
    trainer_model_id: str | None,
    holdings: dict[SampleOwner, list[int]],
    replays_samples: bool,
    reason: Literal["step", "save", "final"],
    rank_weight_witness_supported: bool = True,
    checkpoint_ids: dict[str, str] | None = None,
) -> None:
    if not is_event_logger_initialized():
        return

    get_event_logger().log(
        RolloutHoldingsSnapshotEvent,
        dict(
            lineage_id=_lineage_id,
            rollout_id=rollout_id,
            trainer_model_id=trainer_model_id,
            holdings=holdings,
            replays_samples=replays_samples,
            reason=reason,
            rank_weight_witness_supported=rank_weight_witness_supported,
            checkpoint_ids=checkpoint_ids or {},
        ),
        print_log=False,
    )


def checkpoint_ids_of(*, event_dir: Path | None, rollout_id: int, trainer_model_id: str | None) -> dict[str, str]:
    if event_dir is None:
        return {}
    events = [
        event
        for event in read_events(event_dir)
        if isinstance(event, TrainerCheckpointEvent)
        and isinstance(event.source, TrainerControllerProcessIdentity)
        and event.source.model_id == trainer_model_id
        and event.rollout_id == rollout_id
    ]
    return {event.role: event.checkpoint_id for event in sorted(events, key=lambda event: event.timestamp)}


def log_state_restore(
    rollout_id: int | None, *, rollout_ids: dict[str, int] | None, parent_lineage_id: str | None
) -> None:
    if not is_event_logger_initialized():
        return

    get_event_logger().log(
        RolloutStateRestoreEvent,
        dict(
            rollout_id=rollout_id,
            rollout_ids=rollout_ids,
            lineage_id=_lineage_id,
            parent_lineage_id=parent_lineage_id,
        ),
    )


def log_trained_samples(
    *, rollout_id: int, trainer_model_id: str | None, sample_indices: list[int], lineage_id: str | None
) -> None:
    if not is_event_logger_initialized():
        return

    get_event_logger().log(
        TrainerTrainedSamplesEvent,
        dict(
            lineage_id=lineage_id,
            rollout_id=rollout_id,
            trainer_model_id=trainer_model_id,
            sample_indices=sample_indices,
        ),
        print_log=False,
    )
