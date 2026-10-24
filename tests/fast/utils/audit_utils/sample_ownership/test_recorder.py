from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.fast.import_isolation_utils import modules_imported_by

import miles.utils.audit_utils.sample_ownership.recorder as recorder_module
from miles.rollout.data_source import DataSource
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    ExplicitlyDroppedSamplesEvent,
    TrainerModelCompanionInfoEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_ownership.recorder import SampleOwnershipRecorder
from miles.utils.types import Sample, SampleLineage


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


class TestLogDroppedSamples:
    def test_logs_each_source_sample_once_for_compact_rows(self, event_dir: Path) -> None:
        """Dropping compact rows resolves their source sample once."""
        rows = [
            Sample(index=10, lineage=SampleLineage(source_sample_index=4, output_index=0, output_count=1)),
            Sample(index=10, lineage=SampleLineage(source_sample_index=4, output_index=0, output_count=1)),
        ]

        SampleOwnershipRecorder.log_dropped_samples(args=_args(), samples=rows, reason="dynamic_filter")

        [event] = read_events(event_dir)
        assert isinstance(event, ExplicitlyDroppedSamplesEvent)
        assert event.source_sample_indices == [4]
        assert event.reason == "dynamic_filter"

    def test_group_comparison_logs_only_fully_removed_sources(self, event_dir: Path) -> None:
        """A source with any retained compact row is not reported as dropped."""
        first = Sample(index=10, lineage=SampleLineage(source_sample_index=4, output_index=0, output_count=1))
        second = Sample(index=10, lineage=SampleLineage(source_sample_index=4, output_index=0, output_count=1))
        removed = Sample(index=20, lineage=SampleLineage(source_sample_index=8, output_index=0, output_count=1))

        SampleOwnershipRecorder.log_dropped_groups(
            args=_args(), before=[[first, second], [removed]], after=[[second]], reason="trim"
        )

        [event] = read_events(event_dir)
        assert isinstance(event, ExplicitlyDroppedSamplesEvent)
        assert event.source_sample_indices == [8]

    def test_a_filter_mutating_a_group_in_place_still_reports_the_drop(self, event_dir: Path) -> None:
        """A flattened pre-filter snapshot survives a filter that pops from an inner group list."""
        kept = Sample(index=10, lineage=SampleLineage(source_sample_index=4, output_index=0, output_count=1))
        popped = Sample(index=11, lineage=SampleLineage(source_sample_index=8, output_index=0, output_count=1))
        data = [[kept, popped]]

        before_filter = SampleOwnershipRecorder.flatten_samples(data)
        data[0].pop()
        SampleOwnershipRecorder.log_dropped_groups(
            args=_args(), before=before_filter, after=data, reason="rollout_sample_filter"
        )

        [event] = read_events(event_dir)
        assert isinstance(event, ExplicitlyDroppedSamplesEvent)
        assert event.source_sample_indices == [8]
        assert event.reason == "rollout_sample_filter"

    def test_a_disabled_checker_writes_no_drop_event(self, event_dir: Path) -> None:
        """Drop sites stay silent for a run that never turned the checker on."""
        sample = Sample(index=10, lineage=SampleLineage(source_sample_index=4, output_index=0, output_count=1))
        disabled = _args(enabled=False)

        SampleOwnershipRecorder.log_dropped_samples(args=disabled, samples=[sample], reason="dynamic_filter")
        SampleOwnershipRecorder.log_dropped_source_sample_indices(
            args=disabled, source_sample_indices=[4], reason="trim"
        )
        SampleOwnershipRecorder.log_dropped_groups(args=disabled, before=[[sample]], after=[], reason="oversampling")

        assert read_events(event_dir) == []

    def test_delivered_replay_does_not_repeat_terminal_drop_events(self, event_dir: Path) -> None:
        """Replaying a delivered batch preserves the original trim decision exactly once."""
        sample = Sample(index=8)

        SampleOwnershipRecorder.log_dropped_samples(args=_args(), samples=[sample], reason="dp_schedule_trim")

        with SampleOwnershipRecorder.suppress_drop_logging():
            SampleOwnershipRecorder.log_dropped_samples(args=_args(), samples=[sample], reason="dp_schedule_trim")

        events = read_events(event_dir)
        assert len(events) == 1
        assert isinstance(events[0], ExplicitlyDroppedSamplesEvent)


class TestPublishModelCompanionInfo:
    def test_the_step_snapshot_is_appended_to_the_event_log(
        self, event_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One actor step publishes its whole consumption snapshot as one ordinary event."""
        trained = {SampleLineage(source_sample_index=10, output_index=0, output_count=2): 1}
        monkeypatch.setattr(
            recorder_module.ModelCompanionSampleConsumptionUtils,
            "snapshot",
            staticmethod(lambda model, *, is_skipped: {} if is_skipped else trained),
        )

        SampleOwnershipRecorder.publish_model_companion_info([], rollout_id=7, attempt=3, cell_index=2)

        [event] = read_events(event_dir)
        assert isinstance(event, TrainerModelCompanionInfoEvent)
        assert (event.cell_index, event.rollout_id, event.attempt) == (2, 7, 3)
        assert event.sample_counts[0].sample.source_sample_index == 10
        assert event.sample_counts[0].count == 1
        assert event.skipped_nonfinite_sample_counts == []


class TestRecorderImportCycle:
    def test_importing_the_dataset_module_in_a_clean_interpreter_succeeds(self) -> None:
        """The recorder must not pull DataSource at module level, or miles.utils.data cannot import."""
        assert "miles.utils.audit_utils.sample_ownership.recorder" in modules_imported_by("miles.utils.data")
