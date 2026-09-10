from collections.abc import Sequence

import torch

from miles.backends.training_utils.model_companion import ModelCompanionUtils, TrainingSampleIdentity
from miles.utils.audit_utils.event_logger.logger import get_event_logger
from miles.utils.audit_utils.event_logger.models import TrainerCpuWitnessEvent


def make_current_cpu_witness_payload(
    model: Sequence[torch.nn.Module], *, rollout_id: int, cohort_id: str, replica_id: str
) -> dict[str, object]:
    event = get_event_logger().make_event(
        TrainerCpuWitnessEvent,
        {
            "replica_id": replica_id,
            "rollout_id": rollout_id,
            "cohort_id": cohort_id,
            "sample_counts": _snapshot_counts(ModelCompanionUtils.snapshot(model)),
            "skipped_nonfinite_sample_counts": _snapshot_counts(ModelCompanionUtils.snapshot(model, is_skipped=True)),
            "reason": "current",
        },
    )
    return event.model_dump(mode="json")


def _snapshot_counts(counts: dict[TrainingSampleIdentity, int]) -> list[dict[str, object]]:
    return [
        {
            "sample": {
                "source_sample_index": identity.source_sample_index,
                "row_index": identity.row_index,
                "row_count": identity.row_count,
            },
            "count": count,
        }
        for identity, count in sorted(
            counts.items(),
            key=lambda item: (
                item[0].source_sample_index,
                item[0].row_index,
                item[0].row_count,
            ),
        )
    ]
