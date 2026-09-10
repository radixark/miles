from collections.abc import Iterator
from pathlib import Path

import pytest

from miles.rollout.data_source import DataSource
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import DataSourceIssuedSamplesEvent, ExplicitlyDroppedSamplesEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_flow import (
    log_dropped_groups,
    log_dropped_samples,
    record_data_source_issues,
    suppress_drop_logging,
)
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


class TestLogDroppedSamples:
    def test_logs_each_source_sample_once_for_compact_rows(self, event_dir: Path) -> None:
        """Dropping compact rows resolves their source sample once."""
        rows = [
            Sample(index=10, source_sample_index=4),
            Sample(index=10, source_sample_index=4),
        ]

        log_dropped_samples(rows, reason="dynamic_filter", rollout_id=3)

        [event] = read_events(event_dir)
        assert isinstance(event, ExplicitlyDroppedSamplesEvent)
        assert event.sample_indices == [4]
        assert event.reason == "dynamic_filter"
        assert event.rollout_id == 3

    def test_group_comparison_logs_only_fully_removed_sources(self, event_dir: Path) -> None:
        """A source with any retained compact row is not reported as dropped."""
        first = Sample(index=10, source_sample_index=4)
        second = Sample(index=10, source_sample_index=4)
        removed = Sample(index=20, source_sample_index=8)

        log_dropped_groups([[first, second], [removed]], [[second]], reason="trim", rollout_id=5)

        [event] = read_events(event_dir)
        assert isinstance(event, ExplicitlyDroppedSamplesEvent)
        assert event.sample_indices == [8]

    def test_delivered_replay_does_not_repeat_terminal_drop_events(self, event_dir: Path) -> None:
        """Replaying a delivered batch preserves the original trim decision exactly once."""
        sample = Sample(index=8)
        log_dropped_samples([sample], reason="dp_schedule_trim", rollout_id=5)

        with suppress_drop_logging():
            log_dropped_samples([sample], reason="dp_schedule_trim", rollout_id=5)

        events = read_events(event_dir)
        assert len(events) == 1
        assert isinstance(events[0], ExplicitlyDroppedSamplesEvent)
