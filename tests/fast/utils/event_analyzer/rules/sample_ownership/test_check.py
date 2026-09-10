from datetime import datetime, timedelta, timezone

import pytest

from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.check import check
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.models import (
    CurrentTrainerWitnessIssue,
    IssuedSampleIdentityIssue,
    SampleResolutionIssue,
)
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    ExplicitlyDroppedSamplesEvent,
    IssuedSampleGroup,
    TrainerCpuWitnessEvent,
    TrainerWitnessCohortEvent,
    TrainingSampleCount,
    TrainingSampleIdentity,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainProcessIdentity

_ISSUED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
_NOW = _ISSUED_AT + timedelta(minutes=10)
_GRACE = timedelta(minutes=5)
_DATA_SOURCE = SimpleProcessIdentity(component="main")
_TRAINER = TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=0)


def _issued(
    groups: list[tuple[int, list[int]]],
    *,
    timestamp: datetime = _ISSUED_AT,
) -> DataSourceIssuedSamplesEvent:
    return DataSourceIssuedSamplesEvent(
        timestamp=timestamp,
        source=_DATA_SOURCE,
        groups=[
            IssuedSampleGroup(group_index=group_index, sample_indices=sample_indices)
            for group_index, sample_indices in groups
        ],
    )


def _dropped(*sample_indices: int) -> ExplicitlyDroppedSamplesEvent:
    return ExplicitlyDroppedSamplesEvent(
        timestamp=_NOW,
        source=_DATA_SOURCE,
        sample_indices=list(sample_indices),
        reason="filtered",
    )


def _row(sample_index: int, row_index: int = 0, row_count: int = 1, count: int = 1) -> TrainingSampleCount:
    return TrainingSampleCount(
        sample=TrainingSampleIdentity(
            source_sample_index=sample_index,
            row_index=row_index,
            row_count=row_count,
        ),
        count=count,
    )


def _witness(
    rows: list[TrainingSampleCount],
    *,
    replica_id: str = "cell-0",
    rollout_id: int = 1,
    timestamp: datetime = _NOW,
    reason: str = "current",
    cohort_id: str = "cohort-1",
) -> TrainerCpuWitnessEvent:
    return TrainerCpuWitnessEvent(
        timestamp=timestamp,
        source=_TRAINER,
        replica_id=replica_id,
        sample_counts=rows,
        rollout_id=rollout_id,
        reason=reason,
        cohort_id=cohort_id,
    )


def _check(
    events: list[object],
    *,
    witnesses: list[TrainerCpuWitnessEvent] | None = None,
    cohort_ids: list[str] | None = None,
    now: datetime = _NOW,
) -> list[object]:
    if witnesses is not None:
        cohort_id = witnesses[0].cohort_id if witnesses else "cohort-1"
        assert all(event.cohort_id == cohort_id for event in witnesses)
        events = [
            *events,
            *witnesses,
            TrainerWitnessCohortEvent(
                timestamp=_NOW,
                source=_DATA_SOURCE,
                rollout_id=max((event.rollout_id for event in witnesses), default=1),
                replica_ids=cohort_ids or [event.replica_id for event in witnesses],
                cohort_id=cohort_id,
            ),
        ]
    return check(events, grace_period=_GRACE, now=now)


class TestMaturity:
    def test_a_sample_younger_than_the_grace_period_is_not_checked(self) -> None:
        """An in-flight sample remains unresolved until its grace period expires."""
        now = _ISSUED_AT + _GRACE - timedelta(microseconds=1)
        assert _check([_issued([(7, [10])])], witnesses=[_witness([])], now=now) == []

    def test_a_sample_exactly_at_the_grace_boundary_is_checked(self) -> None:
        """The grace boundary itself makes an issued sample eligible for checking."""
        issues = _check([_issued([(7, [10])])], witnesses=[_witness([])], now=_ISSUED_AT + _GRACE)
        assert [(issue.sample_index, issue.replica_id) for issue in issues] == [(10, "cell-0")]

    def test_reissuing_the_same_group_does_not_refresh_its_age(self) -> None:
        """A retry of an identical group preserves the first issuance timestamp."""
        retry = _issued([(7, [10])], timestamp=_ISSUED_AT + timedelta(minutes=9))
        issues = _check([_issued([(7, [10])]), retry], witnesses=[_witness([])])
        assert [issue.sample_index for issue in issues] == [10]

    def test_every_mature_sample_is_checked_again_after_a_later_double_use(self) -> None:
        """A previously valid sample becomes invalid when a later witness reports reuse."""
        events = [_issued([(7, [10])])]
        assert _check(events, witnesses=[_witness([_row(10)])]) == []
        issues = _check(events, witnesses=[_witness([_row(10, count=2)])])
        assert [(issue.sample_index, issue.trained_rows) for issue in issues] == [(10, ["row 0/1: count 2"])]


class TestPerSlotAccounting:
    def test_each_grpo_slot_trained_once_is_complete(self) -> None:
        """Each sample slot resolves independently when every index is trained once."""
        rows = [_row(10), _row(11), _row(12)]
        assert _check([_issued([(7, [10, 11, 12])])], witnesses=[_witness(rows)]) == []

    def test_equal_group_totals_do_not_hide_one_missing_and_one_duplicate_slot(self) -> None:
        """Matching group totals cannot hide opposite errors in individual slots."""
        issues = _check([_issued([(7, [10, 11])])], witnesses=[_witness([_row(10, count=2)])])
        assert [(issue.sample_index, issue.trained_rows) for issue in issues] == [
            (10, ["row 0/1: count 2"]),
            (11, []),
        ]

    def test_a_mature_unresolved_slot_is_reported_with_its_identity(self) -> None:
        """A missing outcome identifies the exact group, slot, and current replica."""
        issues = _check([_issued([(7, [10, 11])])], witnesses=[_witness([_row(10)])])
        assert issues == [
            SampleResolutionIssue(
                description="mature issued sample was not trained",
                group_index=7,
                slot=1,
                sample_index=11,
                replica_id="cell-0",
                trained_rows=[],
                drop_count=0,
            )
        ]

    def test_the_same_sample_index_in_different_slots_is_an_identity_conflict(self) -> None:
        """A reused sample index cannot collapse two distinct GRPO slots into one witness key."""
        issues = _check([_issued([(7, [10, 10])])], witnesses=[_witness([_row(10)])])
        assert issues == [
            IssuedSampleIdentityIssue(
                description="the same sample index was issued for multiple GRPO slots",
                sample_index=10,
                identities=["group 7 slot 0", "group 7 slot 1"],
            )
        ]

    def test_identical_reissued_slots_are_deduplicated(self) -> None:
        """Retrying an identical issuance does not create a second expected consumption."""
        event = _issued([(7, [10, 11])])
        assert _check([event, event.model_copy()], witnesses=[_witness([_row(10), _row(11)])]) == []


class TestCompactRows:
    def test_all_rows_of_one_compacted_sample_trained_once_are_complete(self) -> None:
        """A compacted sample resolves when every declared row appears exactly once."""
        rows = [_row(10, row_index=index, row_count=3) for index in range(3)]
        assert _check([_issued([(7, [10])])], witnesses=[_witness(rows)]) == []

    def test_a_missing_compact_row_is_reported(self) -> None:
        """A compacted sample cannot pass with a gap in its declared row range."""
        rows = [_row(10, row_index=0, row_count=3), _row(10, row_index=2, row_count=3)]
        assert len(_check([_issued([(7, [10])])], witnesses=[_witness(rows)])) == 1

    def test_a_duplicate_compact_row_is_reported(self) -> None:
        """Two witness entries for one compact row are not one complete row set."""
        rows = [_row(10), _row(10)]
        assert len(_check([_issued([(7, [10])])], witnesses=[_witness(rows)])) == 1

    def test_incoherent_row_counts_are_reported(self) -> None:
        """Rows for one source sample must agree on the total compacted row count."""
        rows = [_row(10, row_index=0, row_count=1), _row(10, row_index=1, row_count=2)]
        assert len(_check([_issued([(7, [10])])], witnesses=[_witness(rows)])) == 1

    def test_a_row_trained_twice_is_reported(self) -> None:
        """Every compact row must have a witness count of exactly one."""
        rows = [_row(10, row_index=0, row_count=2), _row(10, row_index=1, row_count=2, count=2)]
        assert len(_check([_issued([(7, [10])])], witnesses=[_witness(rows)])) == 1


class TestExplicitDrops:
    def test_one_explicit_drop_resolves_an_untrained_sample(self) -> None:
        """A sample may resolve through one explicit drop instead of training."""
        assert _check([_issued([(7, [10])]), _dropped(10)], witnesses=[_witness([])]) == []

    def test_a_dropped_sample_with_any_trained_row_is_reported(self) -> None:
        """An explicitly dropped source sample cannot contribute even one compact row."""
        issues = _check([_issued([(7, [10])]), _dropped(10)], witnesses=[_witness([_row(10)])])
        assert [(issue.sample_index, issue.drop_count) for issue in issues] == [(10, 1)]

    def test_duplicate_explicit_drops_are_reported_once(self) -> None:
        """Two explicit drop records do not count as one valid terminal outcome."""
        issues = _check([_issued([(7, [10])]), _dropped(10), _dropped(10)], witnesses=[_witness([])])
        assert [(issue.sample_index, issue.replica_id, issue.drop_count) for issue in issues] == [(10, None, 2)]

    def test_duplicate_indices_inside_one_drop_event_are_reported(self) -> None:
        """One malformed drop payload cannot hide duplicate terminal outcomes."""
        issues = _check([_issued([(7, [10])]), _dropped(10, 10)], witnesses=[_witness([])])
        assert [(issue.sample_index, issue.drop_count) for issue in issues] == [(10, 2)]


class TestCurrentTrainerWitnesses:
    def test_each_current_replica_is_checked_without_adding_replica_counts(self) -> None:
        """Replicated full snapshots each prove one use instead of summing to two uses."""
        witnesses = [_witness([_row(10)], replica_id="cell-0"), _witness([_row(10)], replica_id="cell-1")]
        assert _check([_issued([(7, [10])])], witnesses=witnesses) == []

    def test_one_current_replica_missing_a_mature_sample_is_reported(self) -> None:
        """A complete peer cannot hide a missing mature obligation on another current replica."""
        witnesses = [_witness([_row(10)], replica_id="cell-0"), _witness([], replica_id="cell-1")]
        issues = _check([_issued([(7, [10])])], witnesses=witnesses)
        assert [(issue.sample_index, issue.replica_id) for issue in issues] == [(10, "cell-1")]

    def test_a_retired_replica_from_an_older_rollout_is_not_in_the_current_cohort(self) -> None:
        """A retired FT rank cannot pollute the cohort through its historic final snapshot."""
        retired = _witness([_row(10, count=2)], replica_id="retired-cell", rollout_id=0)
        current = _witness([_row(10)], replica_id="current-cell", rollout_id=1)
        assert _check([_issued([(7, [10])]), retired], witnesses=[current]) == []

    def test_an_unlisted_snapshot_in_the_current_rollout_is_not_a_current_replica(self) -> None:
        """Only replica ids named by the completed cohort marker participate in checking."""
        unlisted = _witness([_row(10, count=2)], replica_id="unlisted-cell")
        current = _witness([_row(10)], replica_id="current-cell")
        assert _check([_issued([(7, [10])]), unlisted], witnesses=[current]) == []

    def test_the_latest_snapshot_for_a_listed_replica_replaces_earlier_state(self) -> None:
        """A current load snapshot replaces an earlier snapshot in the same completed cohort."""
        old = _witness([_row(10, count=2)], timestamp=_NOW - timedelta(minutes=1))
        current = _witness([_row(10)], reason="load")
        assert _check([_issued([(7, [10])]), old], witnesses=[current]) == []

    def test_an_old_snapshot_cannot_fill_a_missing_replica_in_a_new_same_rollout_cohort(self) -> None:
        """A retry cohort cannot borrow a listed replica snapshot from an earlier attempt."""
        old = _witness([_row(10)], replica_id="cell-1", cohort_id="old-cohort")
        current = _witness([_row(10)], replica_id="cell-0", cohort_id="new-cohort")
        issues = _check(
            [_issued([(7, [10])]), old],
            witnesses=[current],
            cohort_ids=["cell-0", "cell-1"],
        )
        assert issues == [
            CurrentTrainerWitnessIssue(
                description="completed trainer witness cohort is missing replica snapshots",
                replicas=["cell-1"],
            )
        ]

    def test_the_latest_cohort_marker_wins_after_restore_to_a_lower_rollout(self) -> None:
        """Cohort selection follows event time when checkpoint restore lowers the rollout id."""
        old = _witness([_row(10, count=2)], replica_id="old-cell", rollout_id=9)
        old_cohort = TrainerWitnessCohortEvent(
            timestamp=_NOW - timedelta(minutes=1),
            source=_DATA_SOURCE,
            rollout_id=9,
            replica_ids=["old-cell"],
            cohort_id="old-cohort",
        )
        restored = _witness(
            [_row(10)],
            replica_id="restored-cell",
            rollout_id=4,
            reason="load",
            cohort_id="restored-cohort",
        )
        assert _check([_issued([(7, [10])]), old, old_cohort], witnesses=[restored]) == []

    def test_a_current_load_snapshot_replaces_historic_success_evidence(self) -> None:
        """A supplied current load snapshot determines recovery state without historic counts."""
        current = _witness([], timestamp=_NOW + timedelta(minutes=1), reason="load")
        issues = _check([_issued([(7, [10])]), _witness([_row(10)])], witnesses=[current])
        assert [(issue.sample_index, issue.replica_id) for issue in issues] == [(10, "cell-0")]

    def test_no_current_witness_for_mature_work_is_reported(self) -> None:
        """Mature obligations cannot pass when no current weight replica was collected."""
        assert _check([_issued([(7, [10])])], witnesses=[], cohort_ids=["cell-0"]) == [
            CurrentTrainerWitnessIssue(
                description="completed trainer witness cohort is missing replica snapshots",
                replicas=["cell-0"],
            )
        ]


class TestValidation:
    def test_a_negative_grace_period_is_rejected(self) -> None:
        """A negative grace period cannot silently make future issuances mature."""
        with pytest.raises(ValueError, match="grace_period must be non-negative"):
            check([], grace_period=-timedelta(seconds=1), now=_NOW)
