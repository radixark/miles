from miles.rollout.data_source import DataSource
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import DataSourceIssuedSamplesEvent, IssuedSampleGroup
from miles.utils.types import Sample


def record_data_source_issues(data_source: DataSource) -> None:
    get_samples = data_source.get_samples

    def get_samples_and_record(num_samples: int) -> list[list[Sample]]:
        groups = get_samples(num_samples)
        _log_issued_groups(groups)
        return groups

    data_source.get_samples = get_samples_and_record


def _log_issued_groups(groups: list[list[Sample]]) -> None:
    if not groups or not is_event_logger_initialized():
        return

    issued_groups = [
        IssuedSampleGroup(
            group_index=_require_identity(group[0].group_index, "group_index"),
            sample_indices=[_require_identity(sample.index, "sample index") for sample in group],
        )
        for group in groups
    ]
    get_event_logger().log(
        DataSourceIssuedSamplesEvent,
        dict(groups=issued_groups),
        print_log=False,
    )


def _require_identity(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"DataSource returned a sample without a {name}")
    return value
