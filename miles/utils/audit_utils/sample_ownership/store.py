from pathlib import Path

from pydantic import TypeAdapter

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    Event,
    ExplicitlyDroppedSamplesEvent,
    TrainerCpuWitnessEvent,
    TrainerWitnessCohortEvent,
    TrainerWitnessCohortSnapshot,
    TrainGroupStepEndEvent,
)
from miles.utils.file_utils import atomic_write_text

_snapshot_adapter = TypeAdapter(list[TrainerCpuWitnessEvent])


class SampleOwnershipEventStore:
    def __init__(self, event_logger: EventLogger) -> None:
        self._event_logger = event_logger

    def read_history(self) -> list[DataSourceIssuedSamplesEvent | ExplicitlyDroppedSamplesEvent]:
        return [
            event
            for event in self._event_logger.read_events_strict()
            if isinstance(event, (DataSourceIssuedSamplesEvent, ExplicitlyDroppedSamplesEvent))
        ]

    @staticmethod
    def write_snapshot(directory: Path, event: TrainerCpuWitnessEvent) -> None:
        path = _snapshot_path(directory=directory, replica_id=event.replica_id)
        previous = _read_snapshots(path)
        retained = next((item for item in previous if item.rollout_id != event.rollout_id), None)
        snapshots = [event] if retained is None else [event, retained]
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, _snapshot_adapter.dump_json(snapshots).decode())

    def read_current(self) -> TrainerWitnessCohortSnapshot | None:
        for _ in range(3):
            published = [
                event
                for path in (self._event_logger.log_dir / "sample_ownership_current").glob("*.json")
                for event in _read_snapshots(path)
            ]
            if not published:
                return None
            steps = [
                event
                for event in self._event_logger.read_events_strict()
                if isinstance(event, TrainGroupStepEndEvent) and event.role == "actor"
            ]
            if not steps:
                return None
            step = max(enumerate(steps), key=lambda item: (item[1].timestamp, item[0]))[1]
            expected_ids = {
                identity for identities in step.sample_ownership_snapshot_ids.values() for identity in identities
            }
            published_ids = {event.snapshot_id for event in published}
            if not expected_ids.intersection(published_ids):
                completed_ids = {
                    identity
                    for completed_step in steps
                    for identities in completed_step.sample_ownership_snapshot_ids.values()
                    for identity in identities
                }
                if not completed_ids.intersection(published_ids):
                    return None
                continue
            replica_ids = [
                f"cell-{cell_index}"
                for cell_index, outcomes in sorted(step.cell_outcomes.items())
                if outcomes != "error" and outcomes and all(outcome == TrainStepOutcome.NORMAL for outcome in outcomes)
            ]
            assert replica_ids, "Completed training step contains no successful replicas"
            snapshots = []
            for replica_id in replica_ids:
                [snapshot_id] = step.sample_ownership_snapshot_ids[int(replica_id.removeprefix("cell-"))]
                candidates = _read_snapshots(
                    _snapshot_path(directory=self._event_logger.log_dir, replica_id=replica_id)
                )
                matching = [
                    event
                    for event in candidates
                    if event.replica_id == replica_id
                    and event.rollout_id == step.rollout_id
                    and event.attempt == step.attempt
                    and event.cohort_id == f"{step.rollout_id}:{step.attempt}"
                    and event.snapshot_id == snapshot_id
                ]
                if not matching:
                    break
                [snapshot] = matching
                snapshots.append(snapshot)
            if len(snapshots) != len(replica_ids):
                continue
            cutoffs = [event.mature_before for event in snapshots]
            marker = TrainerWitnessCohortEvent(
                timestamp=step.timestamp,
                source=step.source,
                rollout_id=step.rollout_id,
                cohort_id=f"{step.rollout_id}:{step.attempt}",
                replica_ids=replica_ids,
                mature_before=None if any(cutoff is None for cutoff in cutoffs) else min(cutoffs),
            )
            return TrainerWitnessCohortSnapshot(snapshots=snapshots, marker=marker)
        raise RuntimeError("Completed trainer witness cohort is unavailable after three reads")

    def read_events(self) -> list[Event]:
        events: list[Event] = list(self.read_history())
        if (current := self.read_current()) is not None:
            events.extend(current.snapshots)
            events.append(current.marker)
        return events


def _read_snapshots(path: Path) -> list[TrainerCpuWitnessEvent]:
    if not path.is_file():
        return []
    return _snapshot_adapter.validate_json(path.read_text())


def _snapshot_path(*, directory: Path, replica_id: str) -> Path:
    assert replica_id and Path(replica_id).name == replica_id and replica_id not in (".", "..")
    return directory / "sample_ownership_current" / f"{replica_id}.json"
