from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.fast.import_isolation_utils import modules_imported_by

from miles.rollout.data_source import DataSource
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import DataSourceIssuedSamplesEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_ownership.recorder import SampleOwnershipRecorder
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


def _args(*, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(enable_sample_ownership_checker=enabled)


def _install(data_source: DataSource, *, enabled: bool = True, rollout_id: int = 4) -> None:
    SampleOwnershipRecorder.install(
        args=_args(enabled=enabled), data_source=data_source, current_rollout_id=lambda: rollout_id
    )


class TestRecordDataSourceIssues:
    def test_records_group_and_slot_order(self, event_dir: Path) -> None:
        """Issued groups retain their group identity and ordered GRPO sample slots."""
        source = _DataSource()
        _install(source)

        source.get_samples(num_samples=1)

        [event] = read_events(event_dir)
        assert isinstance(event, DataSourceIssuedSamplesEvent)
        assert event.groups[0].group_index == 3
        assert event.groups[0].sample_indices == [10, 11]
        assert event.rollout_id == 4

    def test_stamps_the_rollout_id_the_executor_serves_when_the_samples_are_issued(self, event_dir: Path) -> None:
        """A producer running outside the executor call stamps the rollout served now, not at install time."""
        source = _DataSource()
        served = [0]
        SampleOwnershipRecorder.install(args=_args(), data_source=source, current_rollout_id=lambda: served[0])
        served[0] = 5

        source.get_samples(num_samples=1)

        [event] = read_events(event_dir)
        assert event.rollout_id == 5

    def test_rejects_a_sample_without_an_identity(self, event_dir: Path) -> None:
        """A source cannot silently issue an untrackable sample."""
        source = _DataSource()
        source.get_samples = lambda num_samples: [[Sample(group_index=3, index=None)]]
        _install(source)

        with pytest.raises(ValueError, match="sample index"):
            source.get_samples(num_samples=1)

    def test_a_disabled_checker_leaves_the_data_source_untouched(self, event_dir: Path) -> None:
        """A run without the checker keeps the original get_samples and writes no issuance event."""
        source = _DataSource()
        _install(source, enabled=False)

        source.get_samples(num_samples=1)

        assert "get_samples" not in vars(source)
        assert source.get_samples.__func__ is _DataSource.get_samples
        assert read_events(event_dir) == []


class TestRecorderImportCycle:
    def test_importing_the_dataset_module_in_a_clean_interpreter_succeeds(self) -> None:
        """The recorder must not pull DataSource at module level, or miles.utils.data cannot import."""
        assert "miles.utils.audit_utils.sample_ownership.recorder" in modules_imported_by("miles.utils.data")
