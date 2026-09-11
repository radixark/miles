import uuid
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any

import torch

from miles.backends.training_utils.model_companion import ModelCompanionSampleConsumptionUtils
from miles.rollout.data_source import DataSource
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    ExplicitlyDroppedSamplesEvent,
    IssuedSampleGroup,
    TrainerCpuWitnessEvent,
)
from miles.utils.audit_utils.sample_ownership.store import SampleOwnershipEventStore
from miles.utils.types import Sample, SampleLineage


class SampleOwnershipRecorder:
    @staticmethod
    def install_data_source_issue_recorder(data_source: DataSource) -> None:
        get_samples = data_source.get_samples

        def get_samples_and_record(num_samples: int) -> list[list[Sample]]:
            groups = get_samples(num_samples)
            SampleOwnershipRecorder._log_issued_groups(groups)
            return groups

        data_source.get_samples = get_samples_and_record

    @staticmethod
    def publish_cpu_witness(
        model: Sequence[torch.nn.Module],
        *,
        rollout_id: int,
        attempt: int,
        replica_id: str,
        mature_before: datetime | None = None,
    ) -> str:
        snapshot_id = uuid.uuid4().hex
        event_logger = get_event_logger()
        event = event_logger.make_event(
            TrainerCpuWitnessEvent,
            {
                "replica_id": replica_id,
                "rollout_id": rollout_id,
                "cohort_id": f"{rollout_id}:{attempt}",
                "attempt": attempt,
                "snapshot_id": snapshot_id,
                "mature_before": mature_before,
                "sample_counts": SampleOwnershipRecorder._snapshot_counts(
                    ModelCompanionSampleConsumptionUtils.snapshot(model, is_skipped=False)
                ),
                "skipped_nonfinite_sample_counts": SampleOwnershipRecorder._snapshot_counts(
                    ModelCompanionSampleConsumptionUtils.snapshot(model, is_skipped=True)
                ),
                "reason": "train_end",
            },
        )
        SampleOwnershipEventStore.write_snapshot(directory=event_logger.log_dir, event=event)
        return snapshot_id

    @staticmethod
    def log_dropped_groups(
        before: list[Any],
        after: list[Any],
        *,
        reason: str,
        rollout_id: int | None = None,
    ) -> None:
        retained = {
            SampleOwnershipRecorder._source_sample_index(sample)
            for sample in SampleOwnershipRecorder._iter_samples(after)
        }
        removed = [
            sample
            for sample in SampleOwnershipRecorder._iter_samples(before)
            if SampleOwnershipRecorder._source_sample_index(sample) not in retained
        ]
        SampleOwnershipRecorder.log_dropped_samples(removed, reason=reason, rollout_id=rollout_id)

    @staticmethod
    def log_dropped_samples(samples: list[Sample], *, reason: str, rollout_id: int | None = None) -> None:
        if not samples or not is_event_logger_initialized():
            return

        SampleOwnershipRecorder.log_dropped_sample_indices(
            [SampleOwnershipRecorder._source_sample_index(sample) for sample in samples],
            reason=reason,
            rollout_id=rollout_id,
        )

    @staticmethod
    def log_dropped_sample_indices(
        sample_indices: list[int],
        *,
        reason: str,
        rollout_id: int | None = None,
    ) -> None:
        sample_indices = list(dict.fromkeys(sample_indices))
        if not sample_indices or not is_event_logger_initialized():
            return

        get_event_logger().log(
            ExplicitlyDroppedSamplesEvent,
            dict(sample_indices=sample_indices, reason=reason, rollout_id=rollout_id),
            print_log=False,
        )

    @staticmethod
    def _log_issued_groups(groups: list[list[Sample]]) -> None:
        if not groups or not is_event_logger_initialized():
            return

        issued_groups = [
            IssuedSampleGroup(
                group_index=SampleOwnershipRecorder._require_identity(group[0].group_index, "group_index"),
                sample_indices=[
                    SampleOwnershipRecorder._require_identity(sample.index, "sample index") for sample in group
                ],
            )
            for group in groups
        ]
        get_event_logger().log(
            DataSourceIssuedSamplesEvent,
            dict(groups=issued_groups),
            print_log=False,
        )

    @staticmethod
    def _iter_samples(node: list[Any]) -> Iterator[Sample]:
        for item in node:
            if isinstance(item, Sample):
                yield item
            else:
                yield from SampleOwnershipRecorder._iter_samples(item)

    @staticmethod
    def _source_sample_index(sample: Sample) -> int:
        index = sample.lineage.source_sample_index if sample.lineage is not None else sample.index
        return SampleOwnershipRecorder._require_identity(index, "source sample index")

    @staticmethod
    def _require_identity(value: int | None, name: str) -> int:
        if value is None:
            raise ValueError(f"DataSource returned a sample without a {name}")
        return value

    @staticmethod
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
