from pathlib import Path

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    Event,
    ExplicitlyDroppedSamplesEvent,
    TrainerWitnessCohortPayload,
    TrainerWitnessCohortSnapshot,
)
from miles.utils.file_utils import atomic_write_text

_CURRENT_COHORT_FILENAME = "sample_ownership_current.json"


class SampleOwnershipEventStore:
    def __init__(self, event_logger: EventLogger) -> None:
        self._event_logger = event_logger
        self._current_path = event_logger.log_dir / _CURRENT_COHORT_FILENAME
        self._history: list[DataSourceIssuedSamplesEvent | ExplicitlyDroppedSamplesEvent] = []
        self._history_offset = 0

    @property
    def current_path(self) -> Path:
        return self._current_path

    def read_history(self) -> list[DataSourceIssuedSamplesEvent | ExplicitlyDroppedSamplesEvent]:
        events, self._history_offset = self._event_logger.read_events_strict_from(self._history_offset)
        self._history.extend(
            event
            for event in events
            if isinstance(event, (DataSourceIssuedSamplesEvent, ExplicitlyDroppedSamplesEvent))
        )
        return list(self._history)

    def replace_current(self, payload: TrainerWitnessCohortPayload) -> TrainerWitnessCohortSnapshot:
        snapshot = TrainerWitnessCohortSnapshot.model_validate(payload)
        atomic_write_text(self._current_path, snapshot.model_dump_json())
        return snapshot

    def read_current(self) -> TrainerWitnessCohortSnapshot | None:
        if not self._current_path.is_file():
            return None
        return TrainerWitnessCohortSnapshot.model_validate_json(self._current_path.read_text())

    def read_events(self) -> list[Event]:
        events: list[Event] = list(self.read_history())
        if (current := self.read_current()) is not None:
            events.extend(current.snapshots)
            events.append(current.marker)
        return events

    def reset(self) -> None:
        self._history.clear()
        self._history_offset = 0
