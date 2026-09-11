import argparse
from collections.abc import Callable
from typing import TYPE_CHECKING

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import DataSourceIssuedSamplesEvent, IssuedSampleGroup
from miles.utils.types import Sample

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
    def _require_identity(cls, value: int | None, name: str) -> int:
        if value is None:
            raise ValueError(f"DataSource returned a sample without a {name}")
        return value
