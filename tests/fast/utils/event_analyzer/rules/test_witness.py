"""Tests for event_analyzer rules/witness."""

from datetime import datetime, timezone

from pydantic import TypeAdapter

from miles.backends.megatron_utils.types import TrainStepOutcome
from miles.utils.event_analyzer.rules.witness import WitnessDataMismatchIssue, WitnessMissingSnapshotIssue, check
from miles.utils.event_logger.models import (
    Event,
    TrainAdvantageComputationEvent,
    TrainGroupStepEndEvent,
    WitnessAllocateIdEvent,
    WitnessSnapshotParamEvent,
)
from miles.utils.process_identity import MainProcessIdentity, TrainProcessIdentity

_event_adapter = TypeAdapter(Event)

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_MAIN_SOURCE = MainProcessIdentity()


def _make_source(cell_index: int = 0, rank_within_cell: int = 0) -> TrainProcessIdentity:
    return TrainProcessIdentity(component="actor", cell_index=cell_index, rank_within_cell=rank_within_cell)


def _make_snapshot(
    rollout_id: int,
    nonzero_witness_ids: list[int],
    instance_id: str = "pp0.head",
    cell_index: int = 0,
    rank_within_cell: int = 0,
    stale_ids: list[int] | None = None,
) -> WitnessSnapshotParamEvent:
    return WitnessSnapshotParamEvent(
        timestamp=_FIXED_TS,
        source=_make_source(cell_index=cell_index, rank_within_cell=rank_within_cell),
        rollout_id=rollout_id,
        instance_id=instance_id,
        nonzero_witness_ids=nonzero_witness_ids,
        stale_ids=stale_ids or [],
    )


def _make_allocate(
    rollout_id: int,
    witness_id_to_sample_index: dict[int, int],
    attempt: int = 0,
) -> WitnessAllocateIdEvent:
    return WitnessAllocateIdEvent(
        timestamp=_FIXED_TS,
        source=_MAIN_SOURCE,
        rollout_id=rollout_id,
        attempt=attempt,
        witness_id_to_sample_index=witness_id_to_sample_index,
    )


def _make_step_end(
    rollout_id: int,
    cell_outcomes: dict[int, str | list[TrainStepOutcome]],
) -> TrainGroupStepEndEvent:
    return TrainGroupStepEndEvent(
        timestamp=_FIXED_TS,
        source=_MAIN_SOURCE,
        rollout_id=rollout_id,
        cell_outcomes=cell_outcomes,
    )


def _make_advantage(
    rollout_id: int,
    advantages: list[list[float]],
    witness_ids: list[list[int]],
    cell_index: int = 0,
    attempt: int = 0,
) -> TrainAdvantageComputationEvent:
    return TrainAdvantageComputationEvent(
        timestamp=_FIXED_TS,
        source=_make_source(cell_index=cell_index),
        rollout_id=rollout_id,
        attempt=attempt,
        advantages=advantages,
        witness_ids=witness_ids,
    )


class TestWitnessCheck:
    def test_empty_events(self) -> None:
        assert check([]) == []

    def test_normal_step_with_correct_cumulative_witness_ids_returns_no_issues(self) -> None:
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_normal_step_with_missing_witness_id_returns_issue(self) -> None:
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert issues[0].rollout_id == 0
        assert 11 in issues[0].expected_witness_ids
        assert 11 not in issues[0].actual_witness_ids

    def test_normal_step_with_extra_witness_id_returns_issue(self) -> None:
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11, 99]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert 99 in issues[0].actual_witness_ids
        assert 99 not in issues[0].expected_witness_ids

    def test_discarded_step_is_ignored(self) -> None:
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.DISCARDED_SHOULD_RETRY]}),
        ]
        assert check(events) == []

    def test_stale_ids_are_ignored(self) -> None:
        """IDs in stale_ids are ignored in both expected and actual."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={2: 0, 5: 1, 8: 2}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[5, 8], stale_ids=[0, 1, 2]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_multiple_cells_independent_checking(self) -> None:
        """Each cell is checked independently."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11], cell_index=0),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11], cell_index=1),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_retry_uses_latest_attempt_allocation(self) -> None:
        """When retries happen, only the latest attempt's allocation is used."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}, attempt=0),
            _make_allocate(rollout_id=0, witness_id_to_sample_index={20: 0, 21: 1}, attempt=1),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[20, 21]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_cumulative_across_rollouts(self) -> None:
        """Expected witness IDs are cumulative from rollout 0 to current."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={11: 1}),
            _make_snapshot(rollout_id=1, nonzero_witness_ids=[10, 11]),
            _make_step_end(rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_missing_snapshot_for_normal_cell_returns_issue(self) -> None:
        """Cell claims NORMAL but has no WitnessSnapshotParamEvent — should return WitnessMissingSnapshotIssue."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0}),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert isinstance(issues[0], WitnessMissingSnapshotIssue)
        assert issues[0].rollout_id == 0
        assert issues[0].cell_index == 0

    def test_error_cell_outcome_is_skipped(self) -> None:
        """cell_outcomes with 'error' string should not produce any issue."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0}),
            _make_step_end(rollout_id=0, cell_outcomes={0: "error"}),
        ]
        assert check(events) == []

    def test_multiple_snapshots_per_cell(self) -> None:
        """Same cell has head and tail snapshots; only the mismatched one produces an issue."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11], instance_id="pp0.head"),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10], instance_id="pp0.tail"),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert isinstance(issues[0], WitnessDataMismatchIssue)
        assert 11 in issues[0].expected_witness_ids
        assert 11 not in issues[0].actual_witness_ids

    def test_ring_buffer_wrap_with_stale_ids(self) -> None:
        """After wrap, stale_ids contain wrapped IDs (e.g. [8,9,0]). These should be excluded from comparison."""
        # buffer_size=10, allocated IDs 0..7 in rollout 0, then 8,9,0,1,2 in rollout 1 (wrap)
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={i: i for i in range(8)}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=list(range(8))),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={8: 8, 9: 9, 0: 10, 1: 11, 2: 12}),
            # After wrap: stale_ids=[3,4,5] (old IDs cleaned), actual nonzero = [0,1,2,6,7,8,9]
            _make_snapshot(
                rollout_id=1,
                nonzero_witness_ids=[0, 1, 2, 6, 7, 8, 9],
                stale_ids=[3, 4, 5],
            ),
            _make_step_end(rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        # expected cumulative = {0..9}, minus stale {3,4,5} = {0,1,2,6,7,8,9} — matches actual
        assert check(events) == []

    def test_ring_buffer_wrap_detects_mismatch(self) -> None:
        """After wrap, a genuinely missing non-stale ID should still be caught."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={i: i for i in range(8)}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=list(range(8))),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={8: 8, 9: 9, 0: 10, 1: 11, 2: 12}),
            # ID 7 is NOT stale but missing from actual — should be caught
            _make_snapshot(
                rollout_id=1,
                nonzero_witness_ids=[0, 1, 2, 6, 8, 9],  # missing 7
                stale_ids=[3, 4, 5],
            ),
            _make_step_end(rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert 7 in issues[0].expected_witness_ids
        assert 7 not in issues[0].actual_witness_ids


class TestWitnessEventSerialization:
    def test_roundtrip(self) -> None:
        event = _make_snapshot(
            rollout_id=5,
            nonzero_witness_ids=[10, 20],
            instance_id="pp0.tail",
            cell_index=1,
            stale_ids=[0, 1, 2],
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, WitnessSnapshotParamEvent)
        assert parsed.rollout_id == 5
        assert parsed.instance_id == "pp0.tail"
        assert parsed.nonzero_witness_ids == [10, 20]
        assert parsed.stale_ids == [0, 1, 2]


class TestZeroAdvantageExclusion:
    def test_zero_advantage_sample_excluded_from_expected(self) -> None:
        """Witness ID with zero advantage should not cause a mismatch when missing from actual."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_advantage(rollout_id=0, advantages=[[0.0, 0.0], [2.0, 3.0]], witness_ids=[[10], [11]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[11]),  # 10 missing but zero-adv
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_zero_advantage_sample_stays_excluded_at_later_rollouts(self) -> None:
        """Regression: a zero-advantage id stays absent from every later snapshot and must stay excused."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_advantage(rollout_id=0, advantages=[[0.5, 0.5], [0.0, 0.0]], witness_ids=[[10], [11]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={12: 0}),
            _make_snapshot(rollout_id=1, nonzero_witness_ids=[10, 12]),
            _make_step_end(rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_zero_advantage_sample_with_nonzero_signal_is_flagged_extra(self) -> None:
        """A zero-advantage id that nevertheless shows a nonzero witness signal is still reported."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_advantage(rollout_id=0, advantages=[[0.5, 0.5], [0.0, 0.0]], witness_ids=[[10], [11]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert isinstance(issues[0], WitnessDataMismatchIssue)
        assert 11 in issues[0].actual_witness_ids
        assert 11 not in issues[0].expected_witness_ids

    def test_zero_advantage_exclusion_uses_final_attempt_only(self) -> None:
        """A crashed attempt's all-zero advantage events must not excuse ids the successful retry trains."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}, attempt=1),
            _make_advantage(rollout_id=0, advantages=[[0.0], [0.0]], witness_ids=[[10], [11]], attempt=0),
            _make_advantage(rollout_id=0, advantages=[[5.0], [0.0]], witness_ids=[[10], [11]], attempt=1),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_nonzero_advantage_sample_still_required(self) -> None:
        """Witness ID with nonzero advantage must still appear — missing produces an issue."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_advantage(rollout_id=0, advantages=[[5.0], [3.0]], witness_ids=[[10], [11]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),  # 11 missing, nonzero adv
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert 11 in issues[0].expected_witness_ids
        assert 11 not in issues[0].actual_witness_ids

    def test_zero_advantage_exclusion_is_global_across_cells(self) -> None:
        """Zero-adv exclusion is unioned across cells: the per-cell snapshot reflects the global
        (allreduced) gradient, so an id observed all-zero on its owning shard is excused for every cell."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            # Cell 0 owns sample 10 and observes zero advantage for it.
            _make_advantage(rollout_id=0, advantages=[[0.0, 0.0], [5.0]], witness_ids=[[10], [11]], cell_index=0),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[11], cell_index=0),
            # Cell 1 never owned sample 10; its snapshot also lacks it and must not be flagged.
            _make_advantage(rollout_id=0, advantages=[[5.0]], witness_ids=[[11]], cell_index=1),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[11], cell_index=1),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_no_advantage_event_means_no_exclusion(self) -> None:
        """Without TrainAdvantageComputationEvent at all, no zero-adv exclusion — missing ID is caught."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
