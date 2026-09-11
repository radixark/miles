import argparse
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import torch

from miles.backends.training_utils.model_companion import ModelCompanionSampleConsumptionUtils
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    ExplicitlyDroppedSamplesEvent,
    IssuedSampleGroup,
    OutputConsumption,
    SampleLineagePayload,
    TrainerModelCompanionInfoEvent,
)
from miles.utils.types import Sample, SampleLineage

if TYPE_CHECKING:
    from miles.rollout.data_source import DataSource


class SampleOwnershipRecorder:
    @classmethod
    def install(
        cls,
        *,
        args: argparse.Namespace,
        data_source: "DataSource",
        current_rollout_id: Callable[[], int],
    ) -> None:
        if not args.enable_sample_ownership_checker:
            return

        get_samples = data_source.get_samples

        def get_samples_and_record(num_samples: int) -> list[list[Sample]]:
            groups = get_samples(num_samples)
            cls._log_issued_groups(args=args, groups=groups, rollout_id=current_rollout_id())
            return groups

        data_source.get_samples = get_samples_and_record

    @classmethod
    def publish_model_companion_info(
        cls,
        model: Sequence[torch.nn.Module],
        *,
        rollout_id: int,
        attempt: int,
        cell_index: int,
    ) -> None:
        get_event_logger().log(
            TrainerModelCompanionInfoEvent,
            dict(
                cell_index=cell_index,
                rollout_id=rollout_id,
                attempt=attempt,
                sample_counts=cls._snapshot_counts(
                    ModelCompanionSampleConsumptionUtils.snapshot(model, is_skipped=False)
                ),
                skipped_nonfinite_sample_counts=cls._snapshot_counts(
                    ModelCompanionSampleConsumptionUtils.snapshot(model, is_skipped=True)
                ),
            ),
            print_log=False,
        )

    @classmethod
    def flatten_samples(cls, data: list[Any]) -> list[Sample]:
        return list(cls._iter_samples(data))

    @classmethod
    def log_dropped_groups(cls, *, args: argparse.Namespace, before: list[Any], after: list[Any], reason: str) -> None:
        if not args.enable_sample_ownership_checker or not before or not is_event_logger_initialized():
            return

        retained = {cls._source_sample_index(sample) for sample in cls._iter_samples(after)}
        removed = [sample for sample in cls._iter_samples(before) if cls._source_sample_index(sample) not in retained]
        cls.log_dropped_samples(args=args, samples=removed, reason=reason)

    @classmethod
    def log_dropped_samples(cls, *, args: argparse.Namespace, samples: list[Sample], reason: str) -> None:
        if not args.enable_sample_ownership_checker or not samples or not is_event_logger_initialized():
            return

        cls.log_dropped_source_sample_indices(
            args=args,
            source_sample_indices=[cls._source_sample_index(sample) for sample in samples],
            reason=reason,
        )

    @classmethod
    def log_dropped_source_sample_indices(
        cls, *, args: argparse.Namespace, source_sample_indices: list[int], reason: str
    ) -> None:
        if not args.enable_sample_ownership_checker or not source_sample_indices or not is_event_logger_initialized():
            return

        source_sample_indices = list(dict.fromkeys(source_sample_indices))
        get_event_logger().log(
            ExplicitlyDroppedSamplesEvent,
            dict(source_sample_indices=source_sample_indices, reason=reason),
            print_log=False,
        )

    @classmethod
    def _log_issued_groups(cls, *, args: argparse.Namespace, groups: list[list[Sample]], rollout_id: int) -> None:
        if not args.enable_sample_ownership_checker or not groups or not is_event_logger_initialized():
            return

        issued_groups = [
            IssuedSampleGroup(
                group_index=cls._require_identity(group[0].group_index, "group_index"),
                sample_indices=[cls._require_identity(sample.index, "sample index") for sample in group],
            )
            for group in groups
        ]
        event_logger = get_event_logger()
        with event_logger.with_context(dict(rollout_id=rollout_id)):
            event_logger.log(
                DataSourceIssuedSamplesEvent,
                dict(groups=issued_groups),
                print_log=False,
            )

    @classmethod
    def _iter_samples(cls, node: list[Any]) -> Iterator[Sample]:
        for item in node:
            if isinstance(item, Sample):
                yield item
            else:
                yield from cls._iter_samples(item)

    @classmethod
    def _source_sample_index(cls, sample: Sample) -> int:
        index = sample.lineage.source_sample_index if sample.lineage is not None else sample.index
        return cls._require_identity(index, "source sample index")

    @classmethod
    def _require_identity(cls, value: int | None, name: str) -> int:
        if value is None:
            raise ValueError(f"DataSource returned a sample without a {name}")
        return value

    @classmethod
    def _snapshot_counts(cls, counts: dict[SampleLineage, int]) -> list[OutputConsumption]:
        return [
            OutputConsumption(
                sample=SampleLineagePayload(
                    source_sample_index=identity.source_sample_index,
                    output_index=identity.output_index,
                    output_count=identity.output_count,
                ),
                count=count,
            )
            for identity, count in sorted(
                counts.items(),
                key=lambda item: (
                    item[0].source_sample_index,
                    item[0].output_index,
                    item[0].output_count,
                ),
            )
        ]
