from pathlib import Path

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    Event,
    ExplicitlyDroppedSamplesEvent,
    TrainerCpuWitnessEvent,
    TrainerWitnessCohortEvent,
    TrainerWitnessCohortSnapshot,
)
from miles.utils.file_utils import atomic_write_text

_CURRENT_COHORT_FILENAME = "sample_ownership_current.json"


class SampleOwnershipEventStore:
    def __init__(self, event_logger: EventLogger) -> None:
        self._event_logger = event_logger
        self._current_path = event_logger.log_dir / _CURRENT_COHORT_FILENAME

    @property
    def current_path(self) -> Path:
        return self._current_path

    def read_history(self) -> list[DataSourceIssuedSamplesEvent | ExplicitlyDroppedSamplesEvent]:
        return [
            event
            for event in self._event_logger.read_events_strict()
            if isinstance(event, (DataSourceIssuedSamplesEvent, ExplicitlyDroppedSamplesEvent))
        ]

    @staticmethod
    def write_snapshot(directory: Path, event: TrainerCpuWitnessEvent) -> None:
        path = _snapshot_path(directory=directory, replica_id=event.replica_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, event.model_dump_json())

    @staticmethod
    def write_marker(directory: Path, event: TrainerWitnessCohortEvent) -> None:
        atomic_write_text(directory / _CURRENT_COHORT_FILENAME, event.model_dump_json())

    def read_current(self) -> TrainerWitnessCohortSnapshot | None:
        return self.read_current_from(self._event_logger.log_dir)

    @staticmethod
    def read_current_from(directory: Path) -> TrainerWitnessCohortSnapshot | None:
        path = directory / _CURRENT_COHORT_FILENAME
        if not path.is_file():
            return None
        marker = TrainerWitnessCohortEvent.model_validate_json(path.read_text())
        snapshots = [
            TrainerCpuWitnessEvent.model_validate_json(
                _snapshot_path(directory=directory, replica_id=replica_id).read_text()
            )
            for replica_id in marker.replica_ids
        ]
        assert len(set(marker.replica_ids)) == len(marker.replica_ids), "Duplicate witness replicas"
        for replica_id, snapshot in zip(marker.replica_ids, snapshots, strict=True):
            assert snapshot.replica_id == replica_id, "Witness replica does not match its file"
            assert snapshot.cohort_id == marker.cohort_id, "Witness snapshot belongs to a different cohort"
        return TrainerWitnessCohortSnapshot(snapshots=snapshots, marker=marker)

    def read_events(self) -> list[Event]:
        events: list[Event] = list(self.read_history())
        if (current := self.read_current()) is not None:
            events.extend(current.snapshots)
            events.append(current.marker)
        return events


def _snapshot_path(*, directory: Path, replica_id: str) -> Path:
    assert replica_id and Path(replica_id).name == replica_id and replica_id not in (".", "..")
    return directory / "sample_ownership_current" / f"{replica_id}.json"
