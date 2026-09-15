from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.check import (
    _issued_sources,
    _read_latest_model_companion_info,
    check,
    completed_actor_steps,
)
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.models import (
    IssuedSampleIdentityIssue,
    MissingModelCompanionRecordIssue,
    SampleResolutionIssue,
)
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    ExplicitlyDroppedSamplesEvent,
    IssuedSampleGroup,
    OutputConsumption,
    SampleLineagePayload,
    TrainerModelCompanionInfoEvent,
    TrainGroupStepEndEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainProcessIdentity

_ISSUED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
_NOW = _ISSUED_AT + timedelta(minutes=10)
_EARLIER = _NOW - timedelta(minutes=1)
_LATER = _NOW + timedelta(minutes=1)
_DATA_SOURCE = SimpleProcessIdentity(component="main")
_TRAINER = TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=0)


# ================================== builders ==================================


def _issued(
    groups: list[tuple[int, list[int]]],
    *,
    rollout_id: int = 0,
    timestamp: datetime = _ISSUED_AT,
) -> DataSourceIssuedSamplesEvent:
    return DataSourceIssuedSamplesEvent(
        timestamp=timestamp,
        source=_DATA_SOURCE,
        rollout_id=rollout_id,
        groups=[
            IssuedSampleGroup(group_index=group_index, sample_indices=sample_indices)
            for group_index, sample_indices in groups
        ],
    )


def _dropped(*sample_indices: int, rollout_id: int | None = None) -> ExplicitlyDroppedSamplesEvent:
    return ExplicitlyDroppedSamplesEvent(
        timestamp=_NOW,
        source=_DATA_SOURCE,
        source_sample_indices=list(sample_indices),
        reason="filtered",
        rollout_id=rollout_id,
    )


def _consumption(sample_index: int, output_index: int = 0, output_count: int = 1, count: int = 1) -> OutputConsumption:
    return OutputConsumption(
        sample=SampleLineagePayload(
            source_sample_index=sample_index,
            output_index=output_index,
            output_count=output_count,
        ),
        count=count,
    )


def _witness(
    consumptions: list[OutputConsumption],
    *,
    skipped_consumptions: list[OutputConsumption] | None = None,
    cell_index: int = 0,
    rollout_id: int = 1,
    attempt: int = 0,
    timestamp: datetime = _NOW,
) -> TrainerModelCompanionInfoEvent:
    return TrainerModelCompanionInfoEvent(
        timestamp=timestamp,
        source=_TRAINER,
        cell_index=cell_index,
        sample_counts=consumptions,
        skipped_nonfinite_sample_counts=skipped_consumptions or [],
        rollout_id=rollout_id,
        attempt=attempt,
    )


def _step(
    *,
    rollout_id: int = 1,
    attempt: int = 0,
    cell_indices: list[int],
    timestamp: datetime = _NOW,
    role: Literal["actor", "critic"] = "actor",
) -> TrainGroupStepEndEvent:
    return _step_with_outcomes(
        {cell_index: [TrainStepOutcome.NORMAL] for cell_index in cell_indices},
        rollout_id=rollout_id,
        attempt=attempt,
        timestamp=timestamp,
        role=role,
    )


def _step_with_outcomes(
    cell_outcomes: dict[int, Literal["error"] | list[TrainStepOutcome]],
    *,
    rollout_id: int = 1,
    attempt: int = 0,
    timestamp: datetime = _NOW,
    role: Literal["actor", "critic"] = "actor",
) -> TrainGroupStepEndEvent:
    return TrainGroupStepEndEvent(
        timestamp=timestamp,
        source=_DATA_SOURCE,
        rollout_id=rollout_id,
        attempt=attempt,
        role=role,
        cell_outcomes=cell_outcomes,
    )


def _keys(issues: list[object]) -> list[tuple[int, int | None]]:
    return [(issue.sample_index, issue.cell_index) for issue in issues]


def _missing_records(issues: list[object]) -> list[tuple[int, int, int]]:
    return [
        (issue.cell_index, issue.rollout_id, issue.attempt)
        for issue in issues
        if isinstance(issue, MissingModelCompanionRecordIssue)
    ]


def _check(
    events: list[object],
    *,
    witnesses: list[TrainerModelCompanionInfoEvent] | None = None,
    cell_indices: list[int] | None = None,
    grace_steps: int = 0,
) -> list[object]:
    if witnesses is not None:
        events = [
            *events,
            *witnesses,
            _step(
                rollout_id=max((event.rollout_id for event in witnesses), default=1),
                cell_indices=(cell_indices if cell_indices is not None else [event.cell_index for event in witnesses]),
            ),
        ]
    return check(events, grace_steps=grace_steps)


# ============================= check() scenarios ==============================


class TestCurrentStepSelection:
    def test_no_events_reach_no_verdict(self) -> None:
        """An empty event log has no completed step to check against."""
        assert check([], grace_steps=0) == []

    def test_a_critic_step_alone_reaches_no_verdict(self) -> None:
        """A completed critic step does not establish the actor's current cell set."""
        events = [_issued([(7, [10])]), _witness([]), _step(cell_indices=[0], role="critic")]
        assert check(events, grace_steps=0) == []

    @pytest.mark.parametrize(
        "cell_outcomes",
        [
            {0: [TrainStepOutcome.NORMAL], 1: "error"},
            {0: [TrainStepOutcome.NORMAL, TrainStepOutcome.DISCARDED_SHOULD_RETRY], 1: [TrainStepOutcome.NORMAL]},
            {0: [], 1: [TrainStepOutcome.NORMAL]},
            {},
        ],
    )
    def test_a_step_that_did_not_train_normally_everywhere_reaches_no_verdict(
        self, cell_outcomes: dict[int, Literal["error"] | list[TrainStepOutcome]]
    ) -> None:
        """A step with an error, a discarded outcome, or no outcome on some cell is not a completed step."""
        events = [
            _issued([(7, [10])]),
            _witness([], cell_index=0),
            _witness([], cell_index=1),
            _step_with_outcomes(cell_outcomes),
        ]
        assert check(events, grace_steps=0) == []

    def test_a_failed_later_step_leaves_the_earlier_completed_step_current(self) -> None:
        """A step that failed after a completed one does not hide what the completed one recorded."""
        events = [
            _issued([(7, [10])], rollout_id=0),
            _witness([], rollout_id=1, timestamp=_EARLIER),
            _step(rollout_id=1, cell_indices=[0], timestamp=_EARLIER),
            _witness([_consumption(10)], rollout_id=2),
            _step_with_outcomes({0: "error"}, rollout_id=2),
        ]
        assert _keys(check(events, grace_steps=0)) == [(10, 0)]

    def test_a_latest_step_missing_a_cell_record_does_not_fall_back_to_an_earlier_step(self) -> None:
        """An unpublished cell is reported instead of being judged against the previous step's records."""
        events = [
            _issued([(7, [10])], rollout_id=0),
            _witness([], rollout_id=1, timestamp=_EARLIER),
            _step(rollout_id=1, cell_indices=[0], timestamp=_EARLIER),
            _witness([_consumption(10)], rollout_id=2, cell_index=0),
            _step(rollout_id=2, cell_indices=[0, 1]),
        ]
        assert _missing_records(check(events, grace_steps=0)) == [(1, 2, 0)]

    def test_only_records_of_the_completed_attempt_are_current(self) -> None:
        """Records of an earlier attempt of the same rollout are not the current state."""
        events = [
            _issued([(7, [10])]),
            _witness([_consumption(10, count=2)], attempt=0),
            _witness([_consumption(10)], attempt=1),
            _step(cell_indices=[0], attempt=1),
        ]
        assert check(events, grace_steps=0) == []

    def test_a_completed_attempt_without_records_does_not_use_the_previous_attempt(self) -> None:
        """A retried attempt must publish its own records rather than inherit the previous attempt's."""
        events = [_issued([(7, [10])]), _witness([], attempt=0), _step(cell_indices=[0], attempt=1)]
        assert _missing_records(check(events, grace_steps=0)) == [(0, 1, 1)]

    def test_the_latest_completed_step_is_chosen_by_timestamp_not_by_list_order(self) -> None:
        """Event order in the log cannot override completion time when picking the current step."""
        events = [
            _issued([(7, [10])]),
            _witness([_consumption(10, count=2)], rollout_id=2, timestamp=_LATER),
            _step(rollout_id=2, cell_indices=[0], timestamp=_LATER),
            _witness([_consumption(10)], rollout_id=1),
            _step(rollout_id=1, cell_indices=[0]),
        ]
        assert _keys(check(events, grace_steps=0)) == [(10, 0)]

    def test_a_record_published_before_its_step_end_is_current(self) -> None:
        """A record timestamped before the step end still belongs to that step."""
        events = [_issued([(7, [10])]), _witness([_consumption(10, count=2)], timestamp=_EARLIER)]
        assert _keys(_check(events, witnesses=[], cell_indices=[0])) == [(10, 0)]

    def test_a_duplicated_step_end_does_not_change_the_verdict(self) -> None:
        """Logging one step end twice neither doubles cells nor moves the current step."""
        step = _step(cell_indices=[0])
        events = [_issued([(7, [10])]), _witness([_consumption(10)]), step, step.model_copy()]
        assert check(events, grace_steps=0) == []

    def test_a_record_for_a_cell_unlisted_by_the_step_is_ignored(self) -> None:
        """A stray record of an unlisted cell neither joins the check nor blocks the listed cells' verdict."""
        events = [
            _issued([(7, [10, 11])]),
            _witness([_consumption(10)], cell_index=0),
            _witness([_consumption(10), _consumption(11)], cell_index=9),
            _step(cell_indices=[0]),
        ]
        assert _keys(check(events, grace_steps=0)) == [(11, 0)]


class TestMissingModelCompanionRecords:
    def test_a_cell_record_missing_within_the_grace_is_only_late(self) -> None:
        """A record still in flight must not be reported, or every normal step would fail the check."""
        events = [
            _witness([], rollout_id=1, cell_index=0),
            _step(rollout_id=1, cell_indices=[0, 1]),
        ]
        assert check(events, grace_steps=2) == []

    def test_a_cell_record_still_missing_after_the_grace_is_reported(self) -> None:
        """Separate per-node event directories otherwise leave the whole check silently empty."""
        events = [
            _witness([], rollout_id=1, cell_index=0),
            _step(rollout_id=1, cell_indices=[0, 1]),
            _witness([], rollout_id=3, cell_index=0),
            _step(rollout_id=3, cell_indices=[0], timestamp=_LATER),
        ]
        assert _missing_records(check(events, grace_steps=2)) == [(1, 1, 0)]


class TestMaturity:
    def test_a_source_issued_after_the_latest_completed_step_is_not_mature(self) -> None:
        """A zero grace still cannot mature a source issued under a step that has not completed."""
        assert _check([_issued([(7, [10])], rollout_id=5)], witnesses=[_witness([])], grace_steps=0) == []

    def test_a_consumed_source_issued_after_the_latest_completed_step_keeps_its_identity(self) -> None:
        """Consumption of a not-yet-mature source is checked and reported under its issued slot."""
        issues = _check([_issued([(7, [10])], rollout_id=5)], witnesses=[_witness([_consumption(10, count=2)])])
        assert [(issue.group_index, issue.slot, issue.sample_index) for issue in issues] == [(7, 0, 10)]

    def test_the_earliest_issuance_wins_regardless_of_event_order(self) -> None:
        """A retry logged before the original issuance still ages the source from the original rollout."""
        events = [_issued([(7, [10])], rollout_id=5), _issued([(7, [10])], rollout_id=0)]
        issues = _check(events, witnesses=[_witness([])], grace_steps=1)
        assert [issue.sample_index for issue in issues] == [10]

    @pytest.mark.parametrize(("latest_rollout_id", "expected"), [(4, []), (5, [(10, 0)])])
    def test_a_multi_step_grace_matures_at_the_step_that_spends_it(
        self, latest_rollout_id: int, expected: list[tuple[int, int]]
    ) -> None:
        """Several completed steps consume the grace one by one until the boundary step completes."""
        steps = [
            _step(rollout_id=rollout_id, cell_indices=[0], timestamp=_ISSUED_AT + timedelta(minutes=rollout_id))
            for rollout_id in range(3, latest_rollout_id)
        ]
        events = [_issued([(7, [10])], rollout_id=2), *steps]
        issues = _check(events, witnesses=[_witness([], rollout_id=latest_rollout_id)], grace_steps=3)
        assert _keys(issues) == expected

    @pytest.mark.parametrize("issued", [False, True])
    @pytest.mark.parametrize("skipped", [False, True])
    def test_consumed_samples_are_checked_before_they_mature(self, issued: bool, skipped: bool) -> None:
        """Consumption makes a source eligible without waiting for issuance evidence or the step grace."""
        events = [_issued([(7, [10, 11])], rollout_id=1)] if issued else []
        consumptions = [_consumption(10, count=2)]
        witness = _witness([], skipped_consumptions=consumptions) if skipped else _witness(consumptions)

        issues = _check(events, witnesses=[witness], grace_steps=5)

        assert [issue.sample_index for issue in issues] == [10]
        assert issues[0].group_index == (7 if issued else None)
        assert issues[0].slot == (0 if issued else None)

    def test_consumed_samples_are_rechecked_without_any_issuance_evidence(self) -> None:
        """An unspent step grace still detects reuse in a later current snapshot."""
        assert _check([], witnesses=[_witness([_consumption(10)])], grace_steps=5) == []
        issues = _check([], witnesses=[_witness([_consumption(10, count=2)])], grace_steps=5)
        assert [issue.sample_index for issue in issues] == [10]

    def test_unissued_consumption_must_have_every_output_on_every_cell(self) -> None:
        """Missing issuance metadata cannot hide incomplete outputs or a missing cell outcome."""
        witnesses = [
            _witness([_consumption(10, output_count=2)], cell_index=0),
            _witness([], cell_index=1),
        ]
        issues = _check([], witnesses=witnesses)
        assert [(issue.sample_index, issue.cell_index) for issue in issues] == [(10, 0), (10, 1)]

    def test_one_step_short_of_the_grace_is_not_mature(self) -> None:
        """An in-flight sample remains unresolved until the step grace is fully spent."""
        assert _check([_issued([(7, [10])], rollout_id=1)], witnesses=[_witness([])], grace_steps=1) == []

    def test_exactly_the_grace_in_completed_steps_matures_a_sample(self) -> None:
        """The step that spends the last of the grace makes an issued sample eligible for checking."""
        issues = _check([_issued([(7, [10])], rollout_id=0)], witnesses=[_witness([])], grace_steps=1)
        assert [(issue.sample_index, issue.cell_index) for issue in issues] == [(10, 0)]

    def test_reissuing_the_same_group_does_not_refresh_its_age(self) -> None:
        """A retry of an identical group preserves the rollout id of the first issuance."""
        retry = _issued([(7, [10])], rollout_id=5)
        issues = _check([_issued([(7, [10])], rollout_id=0), retry], witnesses=[_witness([])], grace_steps=1)
        assert [issue.sample_index for issue in issues] == [10]

    def test_zero_grace_matures_every_issued_sample_immediately(self) -> None:
        """An explicit zero grace checks a sample issued under the latest completed step."""
        issues = _check([_issued([(7, [10])], rollout_id=1)], witnesses=[_witness([])], grace_steps=0)
        assert [issue.sample_index for issue in issues] == [10]

    def test_fewer_completed_steps_than_the_grace_defers_checking(self) -> None:
        """A run that has not trained enough steps yet cannot mature a pending sample."""
        assert _check([_issued([(7, [10])], rollout_id=1)], witnesses=[_witness([])], grace_steps=2) == []

    def test_retried_attempts_of_one_rollout_count_as_one_step(self) -> None:
        """Repeated attempts of the same rollout cannot consume the step grace."""
        steps = [
            _step(rollout_id=2, attempt=0, cell_indices=[0], timestamp=_ISSUED_AT + timedelta(minutes=1)),
            _step(rollout_id=2, attempt=1, cell_indices=[0], timestamp=_ISSUED_AT + timedelta(minutes=2)),
        ]

        assert _check([_issued([(7, [10])], rollout_id=1), *steps], witnesses=[_witness([])], grace_steps=2) == []

    def test_restored_issuance_matures_after_the_grace_in_new_steps(self) -> None:
        """Issuance restored with old rollout ids matures once the resumed run completes the step grace."""
        events = [_issued([(7, [10])], rollout_id=4)]

        assert _check(events, witnesses=[_witness([], rollout_id=5)], grace_steps=2) == []

        issues = _check(events, witnesses=[_witness([], rollout_id=6)], grace_steps=2)

        assert [issue.sample_index for issue in issues] == [10]

    def test_every_mature_sample_is_checked_again_after_a_later_double_use(self) -> None:
        """A previously valid sample becomes invalid when a later witness reports reuse."""
        events = [_issued([(7, [10])])]
        assert _check(events, witnesses=[_witness([_consumption(10)])]) == []
        issues = _check(events, witnesses=[_witness([_consumption(10, count=2)])])
        assert [(issue.sample_index, issue.trained_consumptions) for issue in issues] == [
            (10, ["output 0/1: count 2"])
        ]


class TestPerSlotAccounting:
    def test_each_grpo_slot_trained_once_is_complete(self) -> None:
        """Each sample slot resolves independently when every index is trained once."""
        consumptions = [_consumption(10), _consumption(11), _consumption(12)]
        assert _check([_issued([(7, [10, 11, 12])])], witnesses=[_witness(consumptions)]) == []

    def test_a_nonfinite_skipped_slot_is_complete(self) -> None:
        """A consumption consumed and skipped for nonfinite gradients resolves its issued slot."""
        assert (
            _check(
                [_issued([(7, [10])])],
                witnesses=[_witness([], skipped_consumptions=[_consumption(10)])],
            )
            == []
        )

    def test_equal_group_totals_do_not_hide_one_missing_and_one_duplicate_slot(self) -> None:
        """Matching group totals cannot hide opposite errors in individual slots."""
        issues = _check([_issued([(7, [10, 11])])], witnesses=[_witness([_consumption(10, count=2)])])
        assert [(issue.sample_index, issue.trained_consumptions) for issue in issues] == [
            (10, ["output 0/1: count 2"]),
            (11, []),
        ]

    def test_a_mature_unresolved_slot_is_reported_with_its_identity(self) -> None:
        """A missing outcome identifies the exact group, slot, and current cell."""
        issues = _check([_issued([(7, [10, 11])])], witnesses=[_witness([_consumption(10)])])
        assert issues == [
            SampleResolutionIssue(
                description="source sample had no training outcome",
                group_index=7,
                slot=1,
                sample_index=11,
                cell_index=0,
                trained_consumptions=[],
                skipped_consumptions=[],
                drop_count=0,
            )
        ]

    def test_the_same_sample_index_in_different_slots_is_an_identity_conflict(self) -> None:
        """A reused sample index cannot collapse two distinct GRPO slots into one witness key."""
        issues = _check([_issued([(7, [10, 10])])], witnesses=[_witness([_consumption(10)])])
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
        assert _check([event, event.model_copy()], witnesses=[_witness([_consumption(10), _consumption(11)])]) == []

    def test_an_issuance_with_an_empty_group_creates_no_obligation(self) -> None:
        """A group that names no sample indices is not a source to resolve."""
        assert _check([_issued([(7, [])])], witnesses=[_witness([])]) == []

    def test_a_conflicted_index_is_reported_once_and_skips_resolution_checks(self) -> None:
        """An identity conflict replaces per-cell resolution for that index even when it was consumed."""
        witnesses = [_witness([_consumption(10, count=2)], cell_index=0), _witness([], cell_index=1)]
        issues = _check([_issued([(7, [10]), (8, [10])])], witnesses=witnesses)
        assert issues == [
            IssuedSampleIdentityIssue(
                description="the same sample index was issued for multiple GRPO slots",
                sample_index=10,
                identities=["group 7 slot 0", "group 8 slot 0"],
            )
        ]

    def test_identity_issues_precede_resolution_issues(self) -> None:
        """A conflicted index is reported first and does not hide its unresolved sibling."""
        issues = _check([_issued([(7, [10, 10, 12])])], witnesses=[_witness([])])
        assert [type(issue) for issue in issues] == [IssuedSampleIdentityIssue, SampleResolutionIssue]
        assert issues[1].sample_index == 12


class TestCompactRows:
    def test_all_consumptions_of_one_compacted_sample_trained_once_are_complete(self) -> None:
        """A compacted sample resolves when every declared consumption appears exactly once."""
        consumptions = [_consumption(10, output_index=index, output_count=3) for index in range(3)]
        assert _check([_issued([(7, [10])])], witnesses=[_witness(consumptions)]) == []

    def test_a_missing_compact_consumption_is_reported(self) -> None:
        """A compacted sample cannot pass with a gap in its declared consumption range."""
        consumptions = [
            _consumption(10, output_index=0, output_count=3),
            _consumption(10, output_index=2, output_count=3),
        ]
        assert len(_check([_issued([(7, [10])])], witnesses=[_witness(consumptions)])) == 1

    def test_a_duplicate_compact_consumption_is_reported(self) -> None:
        """Two witness entries for one compact consumption are not one complete consumption set."""
        consumptions = [_consumption(10), _consumption(10)]
        assert len(_check([_issued([(7, [10])])], witnesses=[_witness(consumptions)])) == 1

    def test_incoherent_consumption_counts_are_reported(self) -> None:
        """Consumptions for one source sample must agree on the total compacted consumption count."""
        consumptions = [
            _consumption(10, output_index=0, output_count=1),
            _consumption(10, output_index=1, output_count=2),
        ]
        assert len(_check([_issued([(7, [10])])], witnesses=[_witness(consumptions)])) == 1

    def test_a_consumption_trained_twice_is_reported(self) -> None:
        """Every compact consumption must have a witness count of exactly one."""
        consumptions = [
            _consumption(10, output_index=0, output_count=2),
            _consumption(10, output_index=1, output_count=2, count=2),
        ]
        assert len(_check([_issued([(7, [10])])], witnesses=[_witness(consumptions)])) == 1

    def test_compact_consumptions_may_resolve_across_trained_and_skipped_outcomes(self) -> None:
        """Compact siblings consumed in different steps still form one complete source outcome."""
        trained = [_consumption(10, output_index=0, output_count=3), _consumption(10, output_index=2, output_count=3)]
        skipped = [_consumption(10, output_index=1, output_count=3)]

        assert (
            _check(
                [_issued([(7, [10])])],
                witnesses=[_witness(trained, skipped_consumptions=skipped)],
            )
            == []
        )

    def test_the_same_compact_consumption_cannot_be_both_trained_and_skipped(self) -> None:
        """Two terminal outcomes for one physical consumption remain a duplicate even when their kinds differ."""
        issues = _check(
            [_issued([(7, [10])])],
            witnesses=[_witness([_consumption(10)], skipped_consumptions=[_consumption(10)])],
        )

        assert len(issues) == 1
        assert issues[0].trained_consumptions == ["output 0/1: count 1"]
        assert issues[0].skipped_consumptions == ["output 0/1: count 1"]

    def test_a_nonfinite_skip_counted_twice_is_reported(self) -> None:
        """Repeated skip evidence is the same duplicate consumption error as repeated training."""
        issues = _check(
            [_issued([(7, [10])])],
            witnesses=[_witness([], skipped_consumptions=[_consumption(10, count=2)])],
        )

        assert len(issues) == 1
        assert issues[0].skipped_consumptions == ["output 0/1: count 2"]

    def test_trained_and_skipped_consumptions_must_agree_on_compact_shape(self) -> None:
        """Outcome kinds cannot disagree about how many consumptions one source produced."""
        issues = _check(
            [_issued([(7, [10])])],
            witnesses=[
                _witness(
                    [_consumption(10, output_index=0, output_count=2)],
                    skipped_consumptions=[_consumption(10, output_index=1, output_count=3)],
                )
            ],
        )

        assert len(issues) == 1

    def test_one_output_beyond_the_declared_count_is_reported(self) -> None:
        """An extra output index past the declared count is not one complete set."""
        consumptions = [_consumption(10, output_index=index, output_count=2) for index in range(3)]
        assert _keys(_check([_issued([(7, [10])])], witnesses=[_witness(consumptions)])) == [(10, 0)]

    @pytest.mark.parametrize("output_count", [0, -1])
    def test_a_nonpositive_declared_output_count_is_reported(self, output_count: int) -> None:
        """A declared output count of zero or less can never describe a complete set."""
        witness = _witness([_consumption(10, output_count=output_count)])
        assert _keys(_check([_issued([(7, [10])])], witnesses=[witness])) == [(10, 0)]

    @pytest.mark.parametrize("output_index", [1, -1])
    def test_a_single_output_must_be_output_zero(self, output_index: int) -> None:
        """A one-output source is complete only through output index zero."""
        witness = _witness([_consumption(10, output_index=output_index)])
        assert _keys(_check([_issued([(7, [10])])], witnesses=[witness])) == [(10, 0)]

    def test_one_entry_counting_every_output_is_not_a_complete_set(self) -> None:
        """A count summing to the declared output count does not replace one entry per output."""
        consumptions = [_consumption(10, output_index=0, output_count=2, count=2)]
        assert _keys(_check([_issued([(7, [10])])], witnesses=[_witness(consumptions)])) == [(10, 0)]

    def test_every_cell_may_complete_the_set_with_its_own_mix_of_outcomes(self) -> None:
        """Each cell resolves the full set independently, whichever outputs it trained or skipped."""
        outputs = [_consumption(10, output_index=index, output_count=3) for index in range(3)]
        witnesses = [
            _witness(outputs, cell_index=0),
            _witness(outputs[:2], skipped_consumptions=outputs[2:], cell_index=1),
        ]
        assert _check([_issued([(7, [10])])], witnesses=witnesses) == []


class TestExplicitDrops:
    def test_one_explicit_drop_resolves_an_untrained_sample(self) -> None:
        """A sample may resolve through one explicit drop instead of training."""
        assert _check([_issued([(7, [10])]), _dropped(10)], witnesses=[_witness([])]) == []

    def test_a_dropped_sample_with_any_trained_consumption_is_reported(self) -> None:
        """An explicitly dropped source sample cannot contribute even one compact consumption."""
        issues = _check([_issued([(7, [10])]), _dropped(10)], witnesses=[_witness([_consumption(10)])])
        assert [(issue.sample_index, issue.drop_count) for issue in issues] == [(10, 1)]

    def test_a_dropped_sample_with_any_skipped_consumption_is_reported(self) -> None:
        """A whole-source drop is mutually exclusive with nonfinite consumption consumption."""
        issues = _check(
            [_issued([(7, [10])]), _dropped(10)],
            witnesses=[_witness([], skipped_consumptions=[_consumption(10)])],
        )

        assert [(issue.sample_index, issue.drop_count) for issue in issues] == [(10, 1)]

    def test_duplicate_explicit_drops_are_reported_once(self) -> None:
        """Two explicit drop records do not count as one valid terminal outcome."""
        issues = _check([_issued([(7, [10])]), _dropped(10), _dropped(10)], witnesses=[_witness([])])
        assert [(issue.sample_index, issue.cell_index, issue.drop_count) for issue in issues] == [(10, None, 2)]

    def test_duplicate_indices_inside_one_drop_event_are_reported(self) -> None:
        """One malformed drop payload cannot hide duplicate terminal outcomes."""
        issues = _check([_issued([(7, [10])]), _dropped(10, 10)], witnesses=[_witness([])])
        assert [(issue.sample_index, issue.drop_count) for issue in issues] == [(10, 2)]

    def test_a_single_drop_resolves_the_source_on_every_cell(self) -> None:
        """One explicit drop satisfies every current cell that never consumed the source."""
        witnesses = [_witness([], cell_index=0), _witness([], cell_index=1)]
        assert _check([_issued([(7, [10])]), _dropped(10)], witnesses=witnesses) == []

    def test_a_dropped_source_consumed_on_one_cell_is_reported_on_that_cell_only(self) -> None:
        """The drop conflict names the consuming cell and leaves the untouched cell resolved."""
        witnesses = [_witness([_consumption(10)], cell_index=0), _witness([], cell_index=1)]
        issues = _check([_issued([(7, [10])]), _dropped(10)], witnesses=witnesses)
        assert issues == [
            SampleResolutionIssue(
                description="source sample had an output outcome and was explicitly dropped",
                group_index=7,
                slot=0,
                sample_index=10,
                cell_index=0,
                trained_consumptions=["output 0/1: count 1"],
                skipped_consumptions=[],
                drop_count=1,
            )
        ]

    def test_repeated_drops_are_reported_once_across_cells_without_consumption_evidence(self) -> None:
        """Repeated drops are one source-level verdict that takes precedence over per-cell consumptions."""
        witnesses = [
            _witness([_consumption(10)], skipped_consumptions=[_consumption(10)], cell_index=0),
            _witness([], cell_index=1),
        ]
        issues = _check([_issued([(7, [10])]), _dropped(10), _dropped(10)], witnesses=witnesses)
        assert issues == [
            SampleResolutionIssue(
                description="source sample was explicitly dropped more than once",
                group_index=7,
                slot=0,
                sample_index=10,
                cell_index=None,
                trained_consumptions=[],
                skipped_consumptions=[],
                drop_count=2,
            )
        ]

    def test_a_drop_event_naming_no_source_changes_nothing(self) -> None:
        """An empty drop payload neither drops nor resolves anything."""
        issues = _check([_issued([(7, [10])]), _dropped()], witnesses=[_witness([])])
        assert [(issue.sample_index, issue.drop_count) for issue in issues] == [(10, 0)]

    def test_a_drop_stamped_with_a_rollout_id_counts_like_any_other_drop(self) -> None:
        """The rollout id on a drop event is informational and does not gate the drop."""
        events = [_issued([(7, [10])]), _dropped(10, rollout_id=99)]
        assert _check(events, witnesses=[_witness([])]) == []

    def test_dropping_one_sibling_does_not_resolve_the_other(self) -> None:
        """A drop names one source; its group mates still need their own outcome."""
        issues = _check([_issued([(7, [10, 11])]), _dropped(11)], witnesses=[_witness([])])
        assert [(issue.sample_index, issue.drop_count) for issue in issues] == [(10, 0)]

    def test_repeated_drops_of_an_immature_unconsumed_source_wait_for_maturity(self) -> None:
        """A drop does not make a source eligible, so repeated drops surface only once it matures."""
        events = [_issued([(7, [10])], rollout_id=1), _dropped(10), _dropped(10)]
        assert _check(events, witnesses=[_witness([], rollout_id=1)], grace_steps=1) == []

        issues = _check(events, witnesses=[_witness([], rollout_id=2)], grace_steps=1)

        assert [(issue.sample_index, issue.drop_count) for issue in issues] == [(10, 2)]

    def test_a_drop_of_a_never_issued_unconsumed_source_is_not_checked(self) -> None:
        """A drop alone is neither issuance nor consumption, so it creates no eligible source."""
        assert _check([_dropped(99)], witnesses=[_witness([])]) == []

    def test_a_dropped_unissued_source_that_was_consumed_is_reported(self) -> None:
        """Consumption makes an unissued source eligible, and its drop then conflicts with that consumption."""
        issues = _check([_dropped(13)], witnesses=[_witness([_consumption(13)])])
        assert [(issue.sample_index, issue.group_index, issue.drop_count) for issue in issues] == [(13, None, 1)]


class TestCurrentTrainerWitnesses:
    def test_each_current_cell_is_checked_without_adding_cell_counts(self) -> None:
        """Replicated full snapshots each prove one use instead of summing to two uses."""
        witnesses = [_witness([_consumption(10)], cell_index=0), _witness([_consumption(10)], cell_index=1)]
        assert _check([_issued([(7, [10])])], witnesses=witnesses) == []

    def test_one_current_cell_missing_a_mature_sample_is_reported(self) -> None:
        """A complete peer cannot hide a missing mature obligation on another current cell."""
        witnesses = [_witness([_consumption(10)], cell_index=0), _witness([], cell_index=1)]
        issues = _check([_issued([(7, [10])])], witnesses=witnesses)
        assert [(issue.sample_index, issue.cell_index) for issue in issues] == [(10, 1)]

    def test_a_retired_cell_from_an_older_rollout_is_not_a_current_cell(self) -> None:
        """A retired FT rank cannot pollute the current state through its historic final record."""
        retired = _witness([_consumption(10, count=2)], cell_index=1, rollout_id=0)
        current = _witness([_consumption(10)], cell_index=0, rollout_id=1)
        assert _check([_issued([(7, [10])]), retired], witnesses=[current]) == []

    def test_an_unlisted_record_in_the_current_rollout_is_not_a_current_cell(self) -> None:
        """Only cell ids named by the completed step participate in checking."""
        unlisted = _witness([_consumption(10, count=2)], cell_index=9)
        current = _witness([], cell_index=0)
        assert _keys(_check([_issued([(7, [10])]), unlisted], witnesses=[current])) == [(10, 0)]

    def test_two_records_of_one_cell_for_one_attempt_are_rejected(self) -> None:
        """A cell publishing twice for one attempt is a recorder bug, not state to merge or replace."""
        witness = _witness([_consumption(10)])
        with pytest.raises(AssertionError, match=r"rollout 1 attempt 0: \{0: 2\}"):
            _check([_issued([(7, [10])]), witness], witnesses=[witness.model_copy()])

    def test_the_latest_completed_step_wins_after_restore_to_a_lower_rollout(self) -> None:
        """Current state follows event time when checkpoint restore lowers the rollout id."""
        old = _witness([_consumption(10, count=2)], cell_index=1, rollout_id=9)
        old_step = _step(rollout_id=9, cell_indices=[1], timestamp=_NOW - timedelta(minutes=1))
        restored = _witness([_consumption(10)], cell_index=0, rollout_id=4)
        assert _check([_issued([(7, [10])]), old, old_step], witnesses=[restored]) == []

    def test_a_mature_step_whose_cells_published_nothing_is_reported(self) -> None:
        """A step whose cells published no model companion record is reported once the step is mature."""
        assert _check([_issued([(7, [10])])], witnesses=[], cell_indices=[0]) == [
            MissingModelCompanionRecordIssue(
                description="cell published no model companion record for a mature actor step",
                cell_index=0,
                rollout_id=1,
                attempt=0,
            )
        ]

    def test_a_consumption_recorded_under_an_earlier_rollout_does_not_carry_over(self) -> None:
        """Only the current step's record decides; a consumption seen in an earlier rollout is gone."""
        earlier = _witness([_consumption(10)], rollout_id=0, timestamp=_EARLIER)
        issues = _check([_issued([(7, [10])], rollout_id=0), earlier], witnesses=[_witness([], rollout_id=1)])
        assert [(issue.sample_index, issue.cell_index, issue.description) for issue in issues] == [
            (10, 0, "source sample had no training outcome")
        ]

    def test_a_cell_listed_only_by_an_earlier_step_is_not_checked(self) -> None:
        """A cell that left between two completed steps owes nothing for the current step."""
        events = [
            _issued([(7, [10])], rollout_id=0),
            _witness([], cell_index=1, rollout_id=1, timestamp=_EARLIER),
            _witness([_consumption(10)], cell_index=0, rollout_id=1, timestamp=_EARLIER),
            _step(rollout_id=1, cell_indices=[0, 1], timestamp=_EARLIER),
        ]
        assert _check(events, witnesses=[_witness([_consumption(10)], cell_index=0, rollout_id=2)]) == []

    def test_consumption_on_one_cell_makes_an_immature_source_due_on_every_cell(self) -> None:
        """Eligibility through consumption is source-wide, so a silent peer cell is reported."""
        witnesses = [_witness([_consumption(10)], cell_index=0), _witness([], cell_index=1)]
        issues = _check([_issued([(7, [10])], rollout_id=1)], witnesses=witnesses, grace_steps=5)
        assert _keys(issues) == [(10, 1)]

    def test_a_mature_source_with_no_outcome_is_reported_on_every_cell(self) -> None:
        """A source nobody trained or dropped is one issue per current cell."""
        witnesses = [_witness([], cell_index=0), _witness([], cell_index=1)]
        issues = _check([_issued([(7, [10])])], witnesses=witnesses)
        assert issues == [
            SampleResolutionIssue(
                description="source sample had no training outcome",
                group_index=7,
                slot=0,
                sample_index=10,
                cell_index=cell_index,
                trained_consumptions=[],
                skipped_consumptions=[],
                drop_count=0,
            )
            for cell_index in [0, 1]
        ]


class TestValidation:
    def test_a_negative_step_grace_is_rejected(self) -> None:
        """A negative step grace cannot silently make future issuances mature."""
        with pytest.raises(ValueError, match="grace_steps must be non-negative"):
            check([], grace_steps=-1)


class TestOwnershipCoverage:
    def test_one_round_checks_mature_and_consumed_sources_but_not_young_unconsumed_sources(self) -> None:
        """One analysis checks the union of mature issuance and current consumption."""
        events = [
            _issued([(7, [10])], rollout_id=0),
            _issued([(8, [11, 12])], rollout_id=1, timestamp=_NOW),
        ]
        issues = _check(
            events, witnesses=[_witness([_consumption(11, count=2), _consumption(13, count=2)])], grace_steps=1
        )

        assert [(issue.sample_index, issue.group_index, issue.slot) for issue in issues] == [
            (10, 7, 0),
            (11, 8, 0),
            (13, None, None),
        ]
        assert all(isinstance(issue, SampleResolutionIssue) for issue in issues)

    def test_unchanged_events_are_checked_again_when_an_unconsumed_source_matures(self) -> None:
        """An earlier grace exemption does not cache away the later mature obligation."""
        events = [_issued([(7, [10])], rollout_id=1)]
        assert _check(events, witnesses=[_witness([], rollout_id=1)], grace_steps=1) == []

        issues = _check(events, witnesses=[_witness([], rollout_id=2)], grace_steps=1)

        assert [(issue.sample_index, issue.cell_index) for issue in issues] == [(10, 0)]

    def test_a_previously_resolved_drop_is_rechecked_after_a_second_drop(self) -> None:
        """A successful whole-source drop does not exempt later duplicate drop evidence."""
        events = [_issued([(7, [10])]), _dropped(10)]
        assert _check(events, witnesses=[_witness([])]) == []

        issues = _check([*events, _dropped(10)], witnesses=[_witness([])])

        assert [(issue.sample_index, issue.cell_index, issue.drop_count) for issue in issues] == [(10, None, 2)]

    @pytest.mark.parametrize("indices", [(0, 0), (1, 2), (-1, 0)])
    def test_matching_output_totals_cannot_hide_invalid_output_slots(self, indices: tuple[int, int]) -> None:
        """Two outputs must occupy exactly slots zero and one even when total counts match."""
        consumptions = [_consumption(10, output_index=index, output_count=2) for index in indices]
        issues = _check([_issued([(7, [10])])], witnesses=[_witness(consumptions)])

        assert [(issue.sample_index, issue.cell_index) for issue in issues] == [(10, 0)]
        assert isinstance(issues[0], SampleResolutionIssue)

    @pytest.mark.parametrize("count", [0, -1])
    def test_nonpositive_consumption_is_not_a_completed_output(self, count: int) -> None:
        """An observed output needs exactly one consumption rather than a nonpositive count."""
        issues = _check([], witnesses=[_witness([_consumption(10, count=count)])])

        assert [(issue.sample_index, issue.cell_index) for issue in issues] == [(10, 0)]
        assert isinstance(issues[0], SampleResolutionIssue)

    def test_complementary_outputs_on_different_cells_do_not_form_a_complete_sample(self) -> None:
        """Every cell needs the full output set instead of borrowing a peer's output."""
        witnesses = [
            _witness([_consumption(10, output_index=0, output_count=2)], cell_index=0),
            _witness([], skipped_consumptions=[_consumption(10, output_index=1, output_count=2)], cell_index=1),
        ]
        issues = _check([], witnesses=witnesses)

        assert [(issue.sample_index, issue.cell_index) for issue in issues] == [(10, 0), (10, 1)]
        assert all(isinstance(issue, SampleResolutionIssue) for issue in issues)

    def test_a_record_without_a_completed_step_reaches_no_verdict(self) -> None:
        """Uncommitted record evidence cannot silently establish the current cell set."""
        assert _check([_issued([(7, [10])]), _witness([_consumption(10)])]) == []

    def test_a_step_end_naming_no_cell_reaches_no_verdict(self) -> None:
        """A step end naming no cells is not a completed step, so it resolves nothing either way."""
        assert _check([_issued([(7, [10])])], witnesses=[]) == []

    def test_an_incomplete_restored_step_reports_the_cell_that_never_published(self) -> None:
        """A step one of whose cells never published reports that cell instead of borrowing an older step's success."""
        old = _witness([_consumption(10)], cell_index=9, rollout_id=9)
        old_step = _step(rollout_id=9, cell_indices=[9], timestamp=_NOW - timedelta(minutes=1))
        restored = _witness([_consumption(10)], cell_index=0, rollout_id=4)

        issues = _check(
            [_issued([(7, [10])]), old, old_step],
            witnesses=[restored],
            cell_indices=[0, 1],
        )

        assert _missing_records(issues) == [(1, 4, 0)]
        assert [issue for issue in issues if isinstance(issue, SampleResolutionIssue)] == []


class TestIssueReports:
    def test_an_unknown_source_issue_carries_its_consumptions_and_no_identity(self) -> None:
        """A consumed but never issued source is reported with its evidence and empty identity fields."""
        witness = _witness([_consumption(13, output_index=2, output_count=3, count=2)])
        assert _check([], witnesses=[witness]) == [
            SampleResolutionIssue(
                description="source sample does not have one complete set of output outcomes",
                group_index=None,
                slot=None,
                sample_index=13,
                cell_index=0,
                trained_consumptions=["output 2/3: count 2"],
                skipped_consumptions=[],
                drop_count=0,
            )
        ]

    def test_a_drop_conflict_issue_lists_trained_and_skipped_consumptions(self) -> None:
        """The drop conflict report keeps trained and skipped evidence apart."""
        witness = _witness(
            [_consumption(10, output_index=0, output_count=2)],
            skipped_consumptions=[_consumption(10, output_index=1, output_count=2)],
        )
        assert _check([_issued([(7, [10])]), _dropped(10)], witnesses=[witness]) == [
            SampleResolutionIssue(
                description="source sample had an output outcome and was explicitly dropped",
                group_index=7,
                slot=0,
                sample_index=10,
                cell_index=0,
                trained_consumptions=["output 0/2: count 1"],
                skipped_consumptions=["output 1/2: count 1"],
                drop_count=1,
            )
        ]

    def test_issues_are_ordered_by_group_slot_then_cell_then_unknown_index(self) -> None:
        """Issued sources come first in group and slot order, each cell ascending, then unknown sources by index."""
        events = [_issued([(8, [30]), (7, [11, 10])])]
        unknown = [_consumption(21, count=2), _consumption(20, count=2)]
        witnesses = [_witness(unknown, cell_index=1), _witness(unknown, cell_index=0)]

        issues = _check(events, witnesses=witnesses)

        assert [(issue.group_index, issue.slot, issue.sample_index, issue.cell_index) for issue in issues] == [
            (7, 0, 11, 0),
            (7, 0, 11, 1),
            (7, 1, 10, 0),
            (7, 1, 10, 1),
            (8, 0, 30, 0),
            (8, 0, 30, 1),
            (None, None, 20, 0),
            (None, None, 20, 1),
            (None, None, 21, 0),
            (None, None, 21, 1),
        ]

    def test_reversing_the_event_log_yields_the_same_issues(self) -> None:
        """The verdict is a pure function of the events, not of their order or of earlier calls."""
        events = [
            _issued([(7, [10, 11])], rollout_id=0),
            _issued([(7, [10, 11])], rollout_id=3),
            _dropped(11),
            _witness([_consumption(10, count=2)], rollout_id=1, cell_index=1),
            _witness([_consumption(10)], rollout_id=1, cell_index=0),
            _step(rollout_id=1, cell_indices=[0, 1]),
            _witness([_consumption(10)], rollout_id=0, cell_index=0, timestamp=_EARLIER),
            _step(rollout_id=0, cell_indices=[0], timestamp=_EARLIER),
        ]

        issues = check(events, grace_steps=1)

        assert _keys(issues) == [(10, 1)]
        assert check(list(reversed(events)), grace_steps=1) == issues
        assert check(events, grace_steps=1) == issues


# =========================== private stage helpers ============================


class TestCompletedActorSteps:
    @pytest.mark.parametrize(
        ("cell_outcomes", "completed"),
        [
            ({0: [TrainStepOutcome.NORMAL]}, True),
            ({0: [TrainStepOutcome.NORMAL, TrainStepOutcome.NORMAL], 1: [TrainStepOutcome.NORMAL]}, True),
            ({0: [TrainStepOutcome.NORMAL], 1: "error"}, False),
            ({0: [TrainStepOutcome.DISCARDED_SHOULD_RETRY]}, False),
            ({0: [TrainStepOutcome.NORMAL, TrainStepOutcome.DISCARDED_SHOULD_RETRY]}, False),
            ({0: []}, False),
            ({}, False),
        ],
    )
    def test_only_steps_whose_every_cell_trained_normally_are_completed(
        self, cell_outcomes: dict[int, Literal["error"] | list[TrainStepOutcome]], completed: bool
    ) -> None:
        """A step counts only when it lists a cell and every listed cell reports nothing but normal outcomes."""
        assert (completed_actor_steps([_step_with_outcomes(cell_outcomes)]) != []) is completed

    def test_critic_steps_are_never_completed_actor_steps(self) -> None:
        """A critic step end is not an actor step no matter how it went."""
        assert completed_actor_steps([_step(cell_indices=[0], role="critic")]) == []

    def test_steps_are_ordered_by_timestamp_oldest_first(self) -> None:
        """Completed steps come back in completion order regardless of log order."""
        later = _step(rollout_id=2, cell_indices=[0], timestamp=_LATER)
        earlier = _step(rollout_id=1, cell_indices=[0])
        assert [step.rollout_id for step in completed_actor_steps([later, earlier])] == [1, 2]


class TestIssuedSources:
    def test_no_events_yield_no_sources_and_no_issues(self) -> None:
        """Nothing issued means nothing to resolve and nothing to report."""
        assert _issued_sources([]) == ([], [])

    def test_an_empty_group_yields_no_source(self) -> None:
        """A group without sample indices contributes no slot."""
        assert _issued_sources([_issued([(7, [])])]) == ([], [])

    def test_sources_are_sorted_by_group_then_slot(self) -> None:
        """Sources come back in group and slot order, not in issuance order."""
        sources, issues = _issued_sources([_issued([(8, [30]), (7, [11, 10])])])
        assert issues == []
        assert [(one.group_index, one.slot, one.source_sample_index) for one in sources] == [
            (7, 0, 11),
            (7, 1, 10),
            (8, 0, 30),
        ]

    def test_the_same_slot_and_index_keeps_the_smallest_rollout_id(self) -> None:
        """Repeated issuance of one identity ages from its smallest rollout id, whatever the log order."""
        events = [
            _issued([(7, [10])], rollout_id=5),
            _issued([(7, [10])], rollout_id=2),
            _issued([(7, [10])], rollout_id=9),
        ]
        sources, issues = _issued_sources(events)
        assert issues == []
        assert [one.issued_rollout_id for one in sources] == [2]

    def test_reusing_a_group_slot_for_a_new_index_is_two_sources(self) -> None:
        """A slot that later carries a different index is two distinct sources, not a conflict."""
        events = [_issued([(7, [10])], rollout_id=0), _issued([(7, [20])], rollout_id=1)]
        sources, issues = _issued_sources(events)
        assert issues == []
        assert [(one.source_sample_index, one.issued_rollout_id) for one in sources] == [(10, 0), (20, 1)]

    @pytest.mark.parametrize(
        "events",
        [
            [_issued([(7, [10]), (8, [10])])],
            [_issued([(7, [10])], rollout_id=0), _issued([(8, [10])], rollout_id=3)],
        ],
        ids=["one_event", "two_events"],
    )
    def test_an_index_issued_for_two_groups_is_reported_and_excluded(
        self, events: list[DataSourceIssuedSamplesEvent]
    ) -> None:
        """An index seen under two identities is one identity issue and no source."""
        sources, issues = _issued_sources(events)
        assert sources == []
        assert issues == [
            IssuedSampleIdentityIssue(
                description="the same sample index was issued for multiple GRPO slots",
                sample_index=10,
                identities=["group 7 slot 0", "group 8 slot 0"],
            )
        ]

    def test_swapped_indices_inside_one_group_conflict_both_indices(self) -> None:
        """Reissuing a group with its indices swapped conflicts every swapped index."""
        events = [_issued([(7, [10, 11])], rollout_id=0), _issued([(7, [11, 10])], rollout_id=1)]
        sources, issues = _issued_sources(events)
        assert sources == []
        assert [(issue.sample_index, issue.identities) for issue in issues] == [
            (10, ["group 7 slot 0", "group 7 slot 1"]),
            (11, ["group 7 slot 0", "group 7 slot 1"]),
        ]

    def test_a_conflict_does_not_hide_the_other_sources(self) -> None:
        """Only the conflicted index leaves the source list."""
        sources, issues = _issued_sources([_issued([(7, [10, 10, 12])])])
        assert [one.source_sample_index for one in sources] == [12]
        assert [issue.sample_index for issue in issues] == [10]


# ================ companion info read from a written event log ================


_SOURCE = SimpleProcessIdentity(component="rollout_executor")


def _publish(
    event_logger: EventLogger,
    *,
    rollout_id: int,
    attempt: int = 0,
    cell_index: int = 0,
) -> None:
    event_logger.log(
        TrainerModelCompanionInfoEvent,
        dict(
            cell_index=cell_index,
            rollout_id=rollout_id,
            attempt=attempt,
            sample_counts=[
                OutputConsumption(
                    sample=SampleLineagePayload(source_sample_index=10, output_index=0, output_count=1),
                    count=1,
                )
            ],
            skipped_nonfinite_sample_counts=[],
        ),
        print_log=False,
    )


def _complete(
    event_logger: EventLogger,
    *,
    rollout_id: int,
    attempt: int = 0,
    cell_outcomes: dict | None = None,
    role: str = "actor",
) -> None:
    event_logger.log(
        TrainGroupStepEndEvent,
        dict(
            rollout_id=rollout_id,
            attempt=attempt,
            role=role,
            cell_outcomes={0: [TrainStepOutcome.NORMAL]} if cell_outcomes is None else cell_outcomes,
        ),
        print_log=False,
    )


def _read_latest(directory: Path) -> list[TrainerModelCompanionInfoEvent] | None:
    return _read_latest_model_companion_info(read_events(directory, strict=True))


def test_unfinished_next_rollout_preserves_the_completed_step(tmp_path: Path) -> None:
    """A rank finishing the next rollout cannot move the accepted step forward."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _publish(event_logger, rollout_id=1)
    _complete(event_logger, rollout_id=1)
    _publish(event_logger, rollout_id=2)

    [record] = _read_latest(tmp_path)
    assert record.rollout_id == 1


def test_retries_preserve_the_previous_rollout_until_the_new_attempt_completes(tmp_path: Path) -> None:
    """Repeated unaccepted attempts cannot evict the previous completed rollout."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _publish(event_logger, rollout_id=1)
    _complete(event_logger, rollout_id=1)
    for attempt in range(5):
        _publish(event_logger, rollout_id=2, attempt=attempt)
        [record] = _read_latest(tmp_path)
        assert record.rollout_id == 1

    _complete(event_logger, rollout_id=2, attempt=4)

    [record] = _read_latest(tmp_path)
    assert (record.rollout_id, record.attempt) == (2, 4)


def test_a_failed_cell_leaves_the_step_out_of_the_current_record(tmp_path: Path) -> None:
    """A step whose cells did not all train normally is not a record of current state."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _publish(event_logger, rollout_id=1)
    _complete(event_logger, rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: "error"})

    assert _read_latest(tmp_path) is None


def test_a_step_missing_one_cell_record_is_skipped(tmp_path: Path) -> None:
    """A partially published step defers checking instead of checking fewer cells."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _publish(event_logger, rollout_id=1)
    _complete(event_logger, rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: [TrainStepOutcome.NORMAL]})

    assert _read_latest(tmp_path) is None


def test_a_rank_publishing_after_the_step_end_event_is_still_found(tmp_path: Path) -> None:
    """A record written after the step end still counts for that step."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _complete(event_logger, rollout_id=1)
    _publish(event_logger, rollout_id=1)

    [record] = _read_latest(tmp_path)
    assert record.rollout_id == 1


def test_latest_completed_rollout_may_have_a_lower_id_after_restore(tmp_path: Path) -> None:
    """Completion timestamps select a restored lineage rather than the largest rollout ID."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _publish(event_logger, rollout_id=9)
    _complete(event_logger, rollout_id=9)
    _publish(event_logger, rollout_id=4)
    _complete(event_logger, rollout_id=4)

    [record] = _read_latest(tmp_path)
    assert record.rollout_id == 4


def test_critic_completions_do_not_select_actor_records(tmp_path: Path) -> None:
    """Critic progress cannot change the accepted actor model companion record."""
    event_logger = EventLogger(log_dir=tmp_path, source=_SOURCE)
    _publish(event_logger, rollout_id=1)
    _complete(event_logger, rollout_id=1)
    _complete(event_logger, rollout_id=2, role="critic")

    [record] = _read_latest(tmp_path)
    assert record.rollout_id == 1


def test_no_completed_training_step_has_no_current_record(tmp_path: Path) -> None:
    """Initial rollout acquisition does not require a prior training completion."""
    assert _read_latest(tmp_path) is None


def test_malformed_history_is_fatal(tmp_path: Path) -> None:
    """Corrupt accounting history cannot silently remove an issued obligation."""
    (tmp_path / "events.jsonl").write_text('{"type":"data_source_issued_samples"}\n')

    with pytest.raises(ValidationError):
        read_events(tmp_path, strict=True)
