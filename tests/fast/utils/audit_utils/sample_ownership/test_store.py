from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    IssuedSampleGroup,
    TrainerCpuWitnessEvent,
    TrainGroupStepEndEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_ownership.store import SampleOwnershipEventStore

_SOURCE = SimpleProcessIdentity(component="rollout_executor")
_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _publish(
    directory: Path,
    *,
    rollout_id: int,
    attempt: int = 0,
    replica_id: str = "cell-0",
    at: int = 0,
    mature_before: datetime | None = _TIMESTAMP,
) -> None:
    SampleOwnershipEventStore.write_snapshot(
        directory=directory,
        event=TrainerCpuWitnessEvent(
            timestamp=_TIMESTAMP + timedelta(seconds=at),
            source=_SOURCE,
            replica_id=replica_id,
            rollout_id=rollout_id,
            attempt=attempt,
            snapshot_id=f"{rollout_id}:{attempt}:{replica_id}",
            cohort_id=f"{rollout_id}:{attempt}",
            sample_counts=[],
            skipped_nonfinite_sample_counts=[],
            reason="train_end",
            mature_before=mature_before,
        ),
    )


def _complete(
    event_logger: EventLogger,
    *,
    rollout_id: int,
    attempt: int = 0,
    at: int = 1,
    cell_outcomes: dict | None = None,
    role: str = "actor",
) -> None:
    event_logger.log_event(
        TrainGroupStepEndEvent(
            timestamp=_TIMESTAMP + timedelta(seconds=at),
            source=_SOURCE,
            rollout_id=rollout_id,
            attempt=attempt,
            role=role,
            sample_ownership_snapshot_ids={
                index: [f"{rollout_id}:{attempt}:cell-{index}"]
                for index, outcomes in (
                    {0: [TrainStepOutcome.NORMAL]} if cell_outcomes is None else cell_outcomes
                ).items()
                if outcomes != "error"
            },
            cell_outcomes={0: [TrainStepOutcome.NORMAL]} if cell_outcomes is None else cell_outcomes,
        ),
        print_log=False,
    )


def test_unfinished_next_rollout_preserves_the_completed_cohort(tmp_path: Path) -> None:
    """A rank finishing the next rollout cannot replace the accepted read view."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    store = SampleOwnershipEventStore(event_logger)
    event_logger.log(
        DataSourceIssuedSamplesEvent,
        dict(groups=[IssuedSampleGroup(group_index=1, sample_indices=[10])]),
        print_log=False,
    )
    _publish(tmp_path, rollout_id=1)
    _complete(event_logger, rollout_id=1)
    _publish(tmp_path, rollout_id=2, at=2)

    snapshot = store.read_current()
    assert snapshot.marker.rollout_id == 1
    assert len(store.read_history()) == 1
    assert len(snapshot.snapshots) == 1


def test_retries_preserve_the_previous_rollout_until_the_new_attempt_completes(tmp_path: Path) -> None:
    """Repeated unaccepted attempts cannot evict the previous completed rollout."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    store = SampleOwnershipEventStore(event_logger)
    _publish(tmp_path, rollout_id=1)
    _complete(event_logger, rollout_id=1)
    for attempt in range(5):
        _publish(tmp_path, rollout_id=2, attempt=attempt, at=2 + attempt)
        assert store.read_current().marker.rollout_id == 1
    _complete(event_logger, rollout_id=2, attempt=4, at=8)
    assert store.read_current().marker.cohort_id == "2:4"


def test_successful_survivors_exclude_failed_cells(tmp_path: Path) -> None:
    """Accepted FT membership excludes cells that errored after gradient exchange."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _publish(tmp_path, rollout_id=1)
    _complete(event_logger, rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: "error"})
    snapshot = SampleOwnershipEventStore(event_logger).read_current()
    assert snapshot.marker.replica_ids == ["cell-0"]


@pytest.mark.parametrize("wrong_attempt", [False, True])
def test_complete_event_requires_every_matching_replica(tmp_path: Path, wrong_attempt: bool) -> None:
    """A completed cohort cannot mix an old attempt or a missing replica."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _publish(tmp_path, rollout_id=1)
    if wrong_attempt:
        _publish(tmp_path, rollout_id=1, replica_id="cell-1", attempt=1)
    _complete(event_logger, rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: [TrainStepOutcome.NORMAL]})
    with pytest.raises(RuntimeError, match="unavailable"):
        SampleOwnershipEventStore(event_logger).read_current()


def test_rank_clock_ahead_of_controller_does_not_hide_a_completed_snapshot(tmp_path: Path) -> None:
    """Snapshot identity proves publication without comparing clocks across hosts."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _complete(event_logger, rollout_id=1, at=1)
    _publish(tmp_path, rollout_id=1, at=2)
    assert SampleOwnershipEventStore(event_logger).read_current().marker.rollout_id == 1


def test_latest_completed_rollout_may_have_a_lower_id_after_restore(tmp_path: Path) -> None:
    """Completion timestamps select a restored lineage rather than the largest rollout ID."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _publish(tmp_path, rollout_id=9)
    _complete(event_logger, rollout_id=9)
    _publish(tmp_path, rollout_id=4, at=2)
    _complete(event_logger, rollout_id=4, at=3)
    assert SampleOwnershipEventStore(event_logger).read_current().marker.rollout_id == 4


@pytest.mark.parametrize("second_cutoff", [None, _TIMESTAMP - timedelta(seconds=10)])
def test_cohort_uses_the_most_conservative_replica_grace(tmp_path: Path, second_cutoff: datetime | None) -> None:
    """A newly healed or slower replica cannot mature samples prematurely."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _publish(tmp_path, rollout_id=1)
    _publish(tmp_path, rollout_id=1, replica_id="cell-1", mature_before=second_cutoff)
    _complete(event_logger, rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: [TrainStepOutcome.NORMAL]})
    assert SampleOwnershipEventStore(event_logger).read_current().marker.mature_before == second_cutoff


def test_critic_completions_do_not_select_actor_snapshots(tmp_path: Path) -> None:
    """Critic progress cannot change the accepted actor witness cohort."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _publish(tmp_path, rollout_id=1)
    _complete(event_logger, rollout_id=1)
    _complete(event_logger, rollout_id=2, role="critic", at=2)
    assert SampleOwnershipEventStore(event_logger).read_current().marker.rollout_id == 1


def test_no_completed_training_step_has_no_current_cohort(tmp_path: Path) -> None:
    """Initial rollout acquisition does not require a prior training completion."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    assert SampleOwnershipEventStore(event_logger).read_current() is None


def test_malformed_history_is_fatal(tmp_path: Path) -> None:
    """Corrupt accounting history cannot silently remove an issued obligation."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    (tmp_path / "events.jsonl").write_text('{"type":"data_source_issued_samples"}\n')
    with pytest.raises(ValidationError):
        SampleOwnershipEventStore(event_logger).read_history()


def test_malformed_current_snapshot_is_fatal(tmp_path: Path) -> None:
    """A corrupt rank snapshot cannot turn a completed cohort into a pass."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _publish(tmp_path, rollout_id=1)
    _complete(event_logger, rollout_id=1)
    (tmp_path / "sample_ownership_current" / "cell-0.json").write_text('{"snapshots":[]}')
    with pytest.raises(ValidationError):
        SampleOwnershipEventStore(event_logger).read_current()


def test_reader_retries_when_completion_advances_after_its_initial_file_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newer complete cohort triggers another full read rather than a startup skip."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    store = SampleOwnershipEventStore(event_logger)
    _publish(tmp_path, rollout_id=1)
    _complete(event_logger, rollout_id=1)
    read_events = event_logger.read_events_strict
    advanced = False

    def advance_and_read() -> list:
        nonlocal advanced
        if not advanced:
            advanced = True
            _publish(tmp_path, rollout_id=2, at=2)
            _complete(event_logger, rollout_id=2, at=3)
        return read_events()

    monkeypatch.setattr(event_logger, "read_events_strict", advance_and_read)
    assert store.read_current().marker.rollout_id == 2


def test_unaccepted_publications_after_restore_do_not_reuse_checkpoint_completion_tokens(tmp_path: Path) -> None:
    """Old completion metadata cannot acknowledge a new rank publication with the same rollout ID."""
    import json

    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _complete(event_logger, rollout_id=1)
    _publish(tmp_path, rollout_id=1)
    path = tmp_path / "sample_ownership_current" / "cell-0.json"
    snapshots = json.loads(path.read_text())
    snapshots[0]["snapshot_id"] = "new-process-token"
    path.write_text(json.dumps(snapshots))
    assert SampleOwnershipEventStore(event_logger).read_current() is None
