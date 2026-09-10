from collections.abc import Iterator
from pathlib import Path

import pytest

from miles.rollout.data_source import DataSource
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import DataSourceIssuedSamplesEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_flow import record_data_source_issues
from miles.utils.types import Sample


class _DataSource(DataSource):
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        return [[Sample(group_index=3, index=10), Sample(group_index=3, index=11)]][:num_samples]

    def save(self, rollout_id: int) -> None:
        pass

    def load(self, rollout_id: int | None = None) -> None:
        pass


@pytest.fixture
def event_dir(tmp_path: Path) -> Iterator[Path]:
    set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="rollout_executor")))
    yield tmp_path
    set_event_logger(None)


class TestRecordDataSourceIssues:
    def test_records_group_and_slot_order(self, event_dir: Path) -> None:
        """Issued groups retain their group identity and ordered GRPO sample slots."""
        source = _DataSource()
        record_data_source_issues(source)

        source.get_samples(num_samples=1)

        [event] = read_events(event_dir)
        assert isinstance(event, DataSourceIssuedSamplesEvent)
        assert event.groups[0].group_index == 3
        assert event.groups[0].sample_indices == [10, 11]

    def test_rejects_a_sample_without_an_identity(self, event_dir: Path) -> None:
        """A source cannot silently issue an untrackable sample."""
        source = _DataSource()
        source.get_samples = lambda num_samples: [[Sample(group_index=3, index=None)]]
        record_data_source_issues(source)

        with pytest.raises(ValueError, match="sample index"):
            source.get_samples(num_samples=1)
