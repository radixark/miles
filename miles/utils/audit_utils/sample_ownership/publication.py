from collections.abc import Sequence

import torch

from miles.backends.training_utils.model_companion import ModelCompanionSampleConsumptionUtils
from miles.utils.audit_utils.event_logger.logger import get_event_logger
from miles.utils.audit_utils.event_logger.models import TrainerCpuWitnessEvent
from miles.utils.audit_utils.sample_ownership.store import SampleOwnershipEventStore
from miles.utils.types import SampleLineage


def log_current_cpu_witness(
    model: Sequence[torch.nn.Module], *, rollout_id: int, cohort_id: str, replica_id: str
) -> str:
    event_logger = get_event_logger()
    event = event_logger.make_event(
        TrainerCpuWitnessEvent,
        {
            "replica_id": replica_id,
            "rollout_id": rollout_id,
            "cohort_id": cohort_id,
            "sample_counts": _snapshot_counts(ModelCompanionSampleConsumptionUtils.snapshot(model, is_skipped=False)),
            "skipped_nonfinite_sample_counts": _snapshot_counts(
                ModelCompanionSampleConsumptionUtils.snapshot(model, is_skipped=True)
            ),
            "reason": "current",
        },
    )
    SampleOwnershipEventStore.write_snapshot(directory=event_logger.log_dir, event=event)
    return replica_id


def _snapshot_counts(counts: dict[SampleLineage, int]) -> list[dict[str, object]]:
    return [
        {
            "sample": {
                "source_sample_index": identity.source_sample_index,
                "output_index": identity.output_index,
                "output_count": identity.output_count,
            },
            "count": count,
        }
        for identity, count in sorted(
            counts.items(),
            key=lambda item: (
                item[0].source_sample_index,
                item[0].output_index,
                item[0].output_count,
            ),
        )
    ]
