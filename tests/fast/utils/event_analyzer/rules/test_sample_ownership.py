from datetime import datetime, timedelta, timezone
from typing import Literal

import pytest

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_analyzer.analyzer import _partition_by_model_id
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership import (
    SampleConsumedTwiceIssue,
    SampleLostIssue,
    TrainerWeightHistoryIssue,
)
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership import check as _check
from miles.utils.audit_utils.event_logger.models import (
    CellReconfigureEvent,
    RolloutGroupRoutedEvent,
    RolloutHoldingsSnapshotEvent,
    RolloutStateRestoreEvent,
    SampleOwner,
    SampleOwnerTransitionEvent,
    TrainerCheckpointEvent,
    TrainerCpuWitnessEvent,
    TrainerGroupMappingEvent,
    TrainerTrainedSamplesEvent,
    TrainGroupStepEndEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainProcessIdentity
from miles.utils.audit_utils.witness.cpu import CpuWitness

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
SOURCE = SimpleProcessIdentity(component="rollout_executor")


def check(events: list) -> list:
    witnesses: dict[str | None, CpuWitness] = {}
    expanded = []
    for event in events:
        if isinstance(event, TrainerTrainedSamplesEvent):
            source = TrainProcessIdentity(
                component="actor",
                model_id=event.trainer_model_id,
                cell_index=0,
                rank_within_cell=0,
            )
            witness = witnesses.setdefault(
                event.trainer_model_id, CpuWitness(pipeline_rank=0, chunk_index=0, replica_id=(0,))
            )
            fields = dict(
                lineage_id=event.lineage_id,
                trainer_model_id=event.trainer_model_id,
                rollout_id=event.rollout_id,
                step_id=0,
                attempt=0,
                slot=None,
            )
            groups = {index: [index] for index in event.sample_indices}
            witness.commit(dict(**fields, group_indices=list(groups)))
            expanded.extend(
                [
                    TrainerGroupMappingEvent(timestamp=event.timestamp, source=source, **fields, groups=groups),
                    TrainerCpuWitnessEvent(
                        timestamp=event.timestamp,
                        source=source,
                        lineage_id=event.lineage_id,
                        trainer_model_id=event.trainer_model_id,
                        rollout_id=event.rollout_id,
                        records=witness.get_extra_state()["records"],
                    ),
                ]
            )
        elif isinstance(event, RolloutHoldingsSnapshotEvent) and event.trainer_model_id in witnesses:
            source = TrainProcessIdentity(
                component="actor", model_id=event.trainer_model_id, cell_index=0, rank_within_cell=0
            )
            checkpoint_id = f"fixture-{event.timestamp.isoformat()}-{event.trainer_model_id}"
            expanded.append(
                TrainerCpuWitnessEvent(
                    timestamp=event.timestamp,
                    source=source,
                    lineage_id=event.lineage_id,
                    trainer_model_id=event.trainer_model_id,
                    rollout_id=event.rollout_id,
                    records=witnesses[event.trainer_model_id].get_extra_state()["records"],
                    reason="save" if event.reason == "save" else "train_end",
                    checkpoint_id=checkpoint_id,
                )
            )
            if event.reason == "save":
                expanded.append(
                    TrainerCheckpointEvent(
                        timestamp=event.timestamp,
                        source=SOURCE,
                        rollout_id=event.rollout_id,
                        role="actor",
                        checkpoint_id=checkpoint_id,
                        cell_index=0,
                        rank_count=1,
                        alive_cell_indices=[0],
                    )
                )
                event = event.model_copy(update={"checkpoint_ids": {"actor": checkpoint_id}})
            expanded.append(event)
        else:
            expanded.append(event)
    return _check(expanded)


def at(seconds: int) -> datetime:
    return START + timedelta(seconds=seconds)


def transition(
    seconds: int, *, indices: list[int], to_owner: SampleOwner, lineage_id: str | None = None
) -> SampleOwnerTransitionEvent:
    return SampleOwnerTransitionEvent(
        timestamp=at(seconds),
        source=SOURCE,
        sample_indices=indices,
        from_owner=SampleOwner.DATA_SOURCE,
        to_owner=to_owner,
        lineage_id=lineage_id,
    )


def snapshot(
    seconds: int,
    *,
    rollout_id: int,
    holdings: dict[SampleOwner, list[int]],
    reason: Literal["step", "save", "final"] = "step",
    lineage_id: str | None = None,
    replays_samples: bool = False,
    trainer_model_id: str | None = None,
) -> RolloutHoldingsSnapshotEvent:
    return RolloutHoldingsSnapshotEvent(
        timestamp=at(seconds),
        source=SOURCE,
        rollout_id=rollout_id,
        holdings=holdings,
        trainer_model_id=trainer_model_id,
        replays_samples=replays_samples,
        reason=reason,
        lineage_id=lineage_id,
    )


def trained(
    seconds: int,
    *,
    rollout_id: int,
    indices: list[int],
    lineage_id: str | None = None,
    trainer_model_id: str | None = None,
) -> TrainerTrainedSamplesEvent:
    return TrainerTrainedSamplesEvent(
        timestamp=at(seconds),
        source=SOURCE,
        rollout_id=rollout_id,
        sample_indices=indices,
        lineage_id=lineage_id,
        trainer_model_id=trainer_model_id,
    )


def generated(seconds: int, *indices: int) -> SampleOwnerTransitionEvent:
    return transition(seconds, indices=list(indices), to_owner=SampleOwner.IN_FLIGHT)


class TestNothingIsLost:
    def test_out_of_order_logs_keep_the_latest_reset_even_when_rollout_moves_backwards(self) -> None:
        """An older high rollout cannot override a later checkpoint rollback in shuffled logs."""
        fields = dict(lineage_id=None, trainer_model_id=None, rollout_id=0, step_id=0, attempt=0, slot=None)
        events = [
            generated(0, 1),
            TrainerGroupMappingEvent(timestamp=at(1), source=SOURCE, **fields, groups={7: [1]}),
            TrainerCpuWitnessEvent(
                timestamp=at(4),
                source=SOURCE,
                lineage_id=None,
                trainer_model_id=None,
                rollout_id=0,
                records=[],
                reset=True,
            ),
            TrainerCpuWitnessEvent(
                timestamp=at(2),
                source=SOURCE,
                lineage_id=None,
                trainer_model_id=None,
                rollout_id=9,
                records=[dict(**fields, group_indices=[7])],
                reset=True,
            ),
            snapshot(5, rollout_id=0, holdings={}, reason="save"),
        ]
        [issue] = _check(events)
        assert issue.sample_indices == [1]

    def test_an_exited_rank_cannot_hide_loss_after_a_live_rank_rolls_back(self) -> None:
        """Membership changes exclude a dead rank's optimistic witness even with clock skew."""
        fields = dict(lineage_id=None, trainer_model_id=None, rollout_id=0, step_id=0, attempt=0, slot=None)
        dead = TrainProcessIdentity(component="actor", cell_index=1, rank_within_cell=0)
        live = TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=0)
        events = [
            generated(0, 1),
            TrainerGroupMappingEvent(timestamp=at(1), source=dead, **fields, groups={7: [1]}),
            TrainerCpuWitnessEvent(
                timestamp=at(100),
                source=dead,
                lineage_id=None,
                trainer_model_id=None,
                rollout_id=9,
                records=[dict(**fields, group_indices=[7])],
            ),
            TrainerCpuWitnessEvent(
                timestamp=at(2), source=live, lineage_id=None, trainer_model_id=None, rollout_id=0, records=[]
            ),
            CellReconfigureEvent(
                timestamp=at(3),
                source=SOURCE,
                rollout_id=0,
                quorum_id=1,
                src_cell_index=0,
                healed_cell_indices=[],
                alive_cell_indices_after=[0],
                role="actor",
            ),
            snapshot(4, rollout_id=0, holdings={}, reason="final"),
        ]
        [issue] = _check(events)
        assert issue.sample_indices == [1]

    def test_backend_without_rank_evidence_keeps_delivery_checks_and_marks_weight_verification_unavailable(
        self,
    ) -> None:
        """An unsupported backend remains usable without pretending controller acknowledgments prove weights."""
        events = [
            generated(1, 1),
            trained(2, rollout_id=0, indices=[1]),
            snapshot(3, rollout_id=0, holdings={}, reason="save").model_copy(
                update={"rank_weight_witness_supported": False}
            ),
        ]
        assert _check(events) == []

    def test_controller_success_cannot_substitute_for_weight_evidence(self) -> None:
        """A controller success event does not prove that the loaded weights include the samples."""
        events = [
            generated(1, 1),
            trained(2, rollout_id=0, indices=[1]),
            snapshot(3, rollout_id=0, holdings={}, reason="save"),
        ]
        [issue] = _check(events)
        assert isinstance(issue, SampleLostIssue)
        assert issue.sample_indices == [1]

    def test_backend_without_rank_evidence_uses_causal_rollout_boundary_despite_clock_skew(self) -> None:
        """Unsupported backends retain controller delivery evidence without comparing host clocks."""
        events = [
            generated(1, 7, 8),
            trained(100, rollout_id=0, indices=[7]),
            trained(2, rollout_id=1, indices=[8]),
            snapshot(3, rollout_id=0, holdings={}, reason="save").model_copy(
                update={"rank_weight_witness_supported": False}
            ),
        ]
        [issue] = _check(events)
        assert isinstance(issue, SampleLostIssue)
        assert issue.sample_indices == [8]

    def test_incremental_weight_events_accumulate_until_a_rollback_reset(self) -> None:
        """A reset discards the old weight suffix while subsequent deltas retain its prefix."""
        fields = dict(lineage_id=None, trainer_model_id=None, rollout_id=0, step_id=0, attempt=0, slot=None)
        first = dict(**fields, group_indices=[7])
        second = dict(first, step_id=1, group_indices=[8])
        events = [
            generated(0, 1, 2),
            TrainerGroupMappingEvent(timestamp=at(1), source=SOURCE, **fields, groups={7: [1]}),
            TrainerGroupMappingEvent(timestamp=at(1), source=SOURCE, **dict(fields, step_id=1), groups={8: [2]}),
            TrainerCpuWitnessEvent(
                timestamp=at(2),
                source=SOURCE,
                lineage_id=None,
                trainer_model_id=None,
                rollout_id=0,
                records=[first],
                reset=True,
            ),
            TrainerCpuWitnessEvent(
                timestamp=at(3),
                source=SOURCE,
                lineage_id=None,
                trainer_model_id=None,
                rollout_id=0,
                records=[second],
                reset=False,
            ),
            snapshot(4, rollout_id=0, holdings={}, reason="save"),
        ]
        assert _check(events) == []
        events.insert(
            -1,
            TrainerCpuWitnessEvent(
                timestamp=at(3),
                source=SOURCE,
                lineage_id=None,
                trainer_model_id=None,
                rollout_id=0,
                records=[first],
                reset=True,
            ),
        )
        [issue] = _check(events)
        assert issue.sample_indices == [2]

    def test_restored_weight_witness_overrides_previously_reported_training(self) -> None:
        """A rollback that loses training evidence cannot be hidden by earlier success events."""
        fields = dict(lineage_id=None, trainer_model_id=None, rollout_id=0, step_id=0, attempt=0, slot=None)
        events = [
            generated(1, 1),
            TrainerGroupMappingEvent(timestamp=at(2), source=SOURCE, **fields, groups={7: [1]}),
            TrainerCpuWitnessEvent(
                timestamp=at(2),
                source=SOURCE,
                lineage_id=None,
                trainer_model_id=None,
                rollout_id=0,
                records=[dict(**fields, group_indices=[7])],
            ),
            TrainerCpuWitnessEvent(
                timestamp=at(3), source=SOURCE, lineage_id=None, trainer_model_id=None, rollout_id=0, records=[]
            ),
            snapshot(4, rollout_id=0, holdings={}, reason="save"),
        ]
        [issue] = _check(events)
        assert isinstance(issue, SampleLostIssue)
        assert issue.sample_indices == [1]

    @pytest.mark.parametrize("lose_verifier", [False, True])
    def test_routing_only_creates_obligations_for_the_selected_policies(self, lose_verifier: bool) -> None:
        """A solver-only group is valid, but routing a verifier sample creates a real obligation."""
        events = [
            generated(0, 1),
            RolloutGroupRoutedEvent(timestamp=at(1), source=SOURCE, prompt_indices=[1]),
            transition(2, indices=[1], to_owner=SampleOwner.OUTPUT_BUFFER).model_copy(
                update={"trainer_model_id": "solver"}
            ),
            trained(3, rollout_id=0, indices=[1], trainer_model_id="solver"),
            snapshot(5, rollout_id=0, holdings={}, trainer_model_id="solver", reason="save"),
            snapshot(5, rollout_id=0, holdings={}, trainer_model_id="verifier", reason="save"),
        ]
        if lose_verifier:
            events.append(
                transition(2, indices=[1], to_owner=SampleOwner.OUTPUT_BUFFER).model_copy(
                    update={"trainer_model_id": "verifier"}
                )
            )

        issues = [issue for partition in _partition_by_model_id(events) for issue in check(partition)]

        assert len(issues) == int(lose_verifier)
        if lose_verifier:
            assert isinstance(issues[0], SampleLostIssue)
            assert issues[0].sample_indices == [1]

    def test_restored_holdings_that_disappear_are_reported(self) -> None:
        """Restored output belongs to the new lineage and must not silently disappear."""
        events = [
            restore(1, rollout_id=0, lineage_id="b", parent_lineage_id="a"),
            transition(2, indices=[1], to_owner=SampleOwner.OUTPUT_BUFFER, lineage_id="b"),
            snapshot(3, rollout_id=1, holdings={}, reason="save", lineage_id="b"),
        ]

        [issue] = check(events)
        assert isinstance(issue, SampleLostIssue)
        assert issue.sample_indices == [1]

    def test_the_last_step_loss_is_reported_without_a_final_checkpoint(self) -> None:
        """Final snapshots detect a permanent loss even before three runtime snapshots."""
        events = [
            generated(1, 1),
            snapshot(2, rollout_id=0, holdings={}),
            snapshot(3, rollout_id=0, holdings={}, reason="final"),
        ]

        [issue] = check(events)
        assert isinstance(issue, SampleLostIssue)
        assert issue.sample_indices == [1]

    def test_a_sample_held_by_the_buffer_is_accounted_for(self):
        """The buffer reporting what it holds is the whole point of DataBuffer.snapshot."""
        events = [
            generated(1, 1),
            snapshot(2, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: [1]}, reason="save"),
        ]

        assert check(events) == []

    def test_a_sample_that_reached_the_weights_is_accounted_for(self):
        """A trained sample is owned by the weights and by nothing the rollout side still holds."""
        events = [
            generated(1, 1),
            trained(2, rollout_id=0, indices=[1]),
            snapshot(3, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: []}, reason="save"),
        ]

        assert check(events) == []

    def test_a_sample_with_an_explicit_drop_is_accounted_for(self):
        """A dynamic-filter reject is a decision, and the checker must not confuse it with a leak."""
        events = [
            generated(1, 1),
            transition(2, indices=[1], to_owner=SampleOwner.DROPPED),
            snapshot(3, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: []}, reason="save"),
        ]

        assert check(events) == []

    def test_a_sample_nobody_owns_at_a_checkpoint_is_reported(self):
        """This is the failure a checkpoint makes permanent: the resumed run never sees that prompt again."""
        events = [
            generated(1, 1, 2),
            snapshot(2, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: [1]}, reason="save"),
        ]

        [issue] = check(events)

        assert isinstance(issue, SampleLostIssue)
        assert issue.sample_indices == [2]

    def test_one_unaccounted_instant_is_tolerated(self):
        """The snapshots come from several processes, so a sample can be between two of them for a moment."""
        events = [
            generated(1, 1),
            snapshot(2, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: []}),
            snapshot(3, rollout_id=1, holdings={SampleOwner.OUTPUT_BUFFER: [1]}),
            snapshot(4, rollout_id=2, holdings={SampleOwner.OUTPUT_BUFFER: [1]}),
        ]

        assert check(events) == []

    def test_a_sample_unaccounted_at_three_snapshots_in_a_row_is_reported(self):
        """A leak that outlives the lag between processes is a leak."""
        events = [
            generated(1, 1),
            snapshot(2, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: []}),
            snapshot(3, rollout_id=1, holdings={SampleOwner.OUTPUT_BUFFER: []}),
            snapshot(4, rollout_id=2, holdings={SampleOwner.OUTPUT_BUFFER: []}),
        ]

        [issue] = check(events)

        assert issue.sample_indices == [1]

    def test_an_old_snapshot_cannot_hide_loss_in_the_latest_policy_snapshot(self) -> None:
        """Stale holdings cannot account for a sample missing from the policy's latest snapshot."""
        events = [
            generated(1, 1, 2),
            snapshot(2, rollout_id=0, trainer_model_id="solver", holdings={SampleOwner.OUTPUT_BUFFER: [1, 2]}),
            snapshot(
                3, rollout_id=0, trainer_model_id="solver", holdings={SampleOwner.OUTPUT_BUFFER: [2]}, reason="save"
            ),
        ]

        [issue] = check(events)
        assert isinstance(issue, SampleLostIssue)
        assert issue.sample_indices == [1]

    def test_a_run_that_never_snapshotted_is_not_judged(self):
        """Without a snapshot there is no instant to evaluate an owner at, so silence is the only honest answer."""
        assert check([generated(1, 1)]) == []


class TestNothingIsConsumedTwice:
    def test_sibling_indices_trained_within_one_event_are_counted_once(self) -> None:
        """Sibling indices within one event are consumed once."""
        events = [trained(seconds=i, rollout_id=i, indices=[1, 1]) for i in range(1)]
        events.append(snapshot(3, rollout_id=2, holdings={SampleOwner.HANDED_TO_TRAINER: [1]}))

        assert check(events) == []

    def test_sibling_indices_trained_in_separate_events_are_reported(self) -> None:
        """Sibling indices consumed in separate events reveal replay."""
        events = [trained(seconds=i, rollout_id=i, indices=[1, 1]) for i in range(2)]
        events.append(snapshot(3, rollout_id=2, holdings={SampleOwner.HANDED_TO_TRAINER: [1]}))

        [issue] = check(events)
        assert isinstance(issue, SampleConsumedTwiceIssue)
        assert issue.sample_indices == [1]

    def test_sibling_indices_handed_within_one_event_are_counted_once(self) -> None:
        """Sibling indices within one event are consumed once."""
        events = [transition(seconds=i, indices=[1, 1], to_owner=SampleOwner.HANDED_TO_TRAINER) for i in range(1)]
        events.append(snapshot(3, rollout_id=2, holdings={SampleOwner.HANDED_TO_TRAINER: [1]}))

        assert check(events) == []

    def test_sibling_indices_handed_in_separate_events_are_reported(self) -> None:
        """Sibling indices consumed in separate events reveal replay."""
        events = [transition(seconds=i, indices=[1, 1], to_owner=SampleOwner.HANDED_TO_TRAINER) for i in range(2)]
        events.append(snapshot(3, rollout_id=2, holdings={SampleOwner.HANDED_TO_TRAINER: [1]}))

        [issue] = check(events)
        assert isinstance(issue, SampleConsumedTwiceIssue)
        assert issue.sample_indices == [1]

    def test_a_sample_trained_twice_is_reported(self):
        """The same gradient applied twice moves training off the curve the run reports."""
        events = [
            generated(1, 1),
            trained(2, rollout_id=0, indices=[1]),
            trained(3, rollout_id=1, indices=[1]),
            snapshot(4, rollout_id=1, holdings={SampleOwner.OUTPUT_BUFFER: []}),
        ]

        [issue] = check(events)

        assert isinstance(issue, SampleConsumedTwiceIssue)
        assert issue.sample_indices == [1]

    def test_a_sample_handed_to_training_twice_is_reported(self):
        """An executor replay that fires twice would train the same batch twice."""
        events = [
            generated(1, 1),
            transition(2, indices=[1], to_owner=SampleOwner.HANDED_TO_TRAINER),
            transition(3, indices=[1], to_owner=SampleOwner.HANDED_TO_TRAINER),
            trained(4, rollout_id=0, indices=[1]),
            snapshot(5, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: []}),
        ]

        [issue] = check(events)

        assert issue.sample_indices == [1]

    def test_generating_a_sample_twice_is_not_a_violation(self):
        """A retry regenerates a prompt under its own index; only reaching the weights twice is wrong."""
        events = [
            generated(1, 1),
            transition(2, indices=[1], to_owner=SampleOwner.RETRY_BUFFER),
            generated(3, 1),
            trained(4, rollout_id=0, indices=[1]),
            snapshot(5, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: []}, reason="save"),
        ]

        assert check(events) == []

    def test_a_buffer_that_declares_replays_is_only_held_to_not_losing_samples(self):
        """A replay buffer hands the same sample out on purpose, and that is not the invariant to enforce on it."""
        events = [
            generated(1, 1),
            trained(2, rollout_id=0, indices=[1]),
            trained(3, rollout_id=1, indices=[1]),
            snapshot(4, rollout_id=1, holdings={SampleOwner.OUTPUT_BUFFER: []}, replays_samples=True),
        ]

        assert check(events) == []


class TestLineage:
    def test_a_restored_batch_can_be_handed_once_without_counting_initialization(self) -> None:
        """Registering restored holdings is not another handoff to the trainer."""
        initial = transition(2, indices=[1], to_owner=SampleOwner.HANDED_TO_TRAINER, lineage_id="b")
        events = [
            restore(1, rollout_id=0, lineage_id="b", parent_lineage_id="a"),
            initial.model_copy(update={"reason": "restored"}),
            transition(3, indices=[1], to_owner=SampleOwner.HANDED_TO_TRAINER, lineage_id="b"),
            snapshot(4, rollout_id=1, holdings={SampleOwner.HANDED_TO_TRAINER: [1]}, lineage_id="b"),
        ]

        assert check(events) == []

    def test_single_restore_counts_only_training_retained_in_the_weights(self) -> None:
        """A restore retains ancestor training only through its checkpoint step."""
        events = [
            trained(2, rollout_id=0, indices=[1], lineage_id="a"),
            trained(3, rollout_id=1, indices=[2], lineage_id="a"),
            restore(5, rollout_id=0, lineage_id="b", parent_lineage_id="a"),
            trained(6, rollout_id=1, indices=[2], lineage_id="b"),
            snapshot(7, rollout_id=1, holdings={}, reason="save", lineage_id="b"),
        ]
        assert check(events) == []
        events.append(trained(8, rollout_id=2, indices=[1], lineage_id="b"))
        events.append(snapshot(9, rollout_id=2, holdings={}, lineage_id="b"))

        [issue] = check(events)
        assert isinstance(issue, SampleConsumedTwiceIssue)
        assert issue.sample_indices == [1]

    def test_a_follower_retains_training_up_to_its_own_restore_point(self) -> None:
        """A follower ahead of the leader retains its own checkpointed training."""
        events = [
            trained(2, rollout_id=2, indices=[1], lineage_id="a", trainer_model_id="verifier"),
            restore(5, rollout_id=0, lineage_id="b", parent_lineage_id="a", rollout_ids={"solver": 0, "verifier": 2}),
            trained(6, rollout_id=3, indices=[1], lineage_id="b", trainer_model_id="verifier"),
            snapshot(7, rollout_id=3, holdings={}, reason="save", lineage_id="b", trainer_model_id="verifier"),
        ]

        [issue] = check(events)
        assert isinstance(issue, SampleConsumedTwiceIssue)
        assert issue.sample_indices == [1]

    def test_two_restores_exclude_training_on_the_abandoned_branch(self) -> None:
        """Rolling back and restoring again must exclude the abandoned suffix."""
        events = [
            trained(2, rollout_id=0, indices=[1], lineage_id="a"),
            trained(3, rollout_id=1, indices=[2], lineage_id="a"),
            restore(5, rollout_id=0, lineage_id="b", parent_lineage_id="a"),
            trained(6, rollout_id=1, indices=[2], lineage_id="b"),
            restore(8, rollout_id=1, lineage_id="c", parent_lineage_id="b"),
            trained(9, rollout_id=2, indices=[3], lineage_id="c"),
            snapshot(10, rollout_id=2, holdings={}, reason="save", lineage_id="c"),
        ]

        assert check(events) == []
        events.append(trained(11, rollout_id=3, indices=[2], lineage_id="c"))
        events.append(snapshot(12, rollout_id=3, holdings={}, lineage_id="c"))
        [issue] = check(events)
        assert issue.sample_indices == [2]

    def test_clock_skew_does_not_change_current_or_ancestor_training(self) -> None:
        """Trainer timestamps on either side of restore do not determine lineage."""
        events = [
            trained(100, rollout_id=0, indices=[1], lineage_id="a"),
            restore(50, rollout_id=0, lineage_id="b", parent_lineage_id="a"),
            transition(51, indices=[1, 2], to_owner=SampleOwner.RETRY_BUFFER, lineage_id="b"),
            trained(1, rollout_id=1, indices=[2], lineage_id="b"),
            snapshot(52, rollout_id=1, holdings={}, reason="save", lineage_id="b"),
        ]

        assert check(events) == []
        events.append(trained(2, rollout_id=2, indices=[1], lineage_id="b"))
        events.append(snapshot(53, rollout_id=2, holdings={}, reason="save", lineage_id="b"))
        [issue] = check(events)
        assert issue.sample_indices == [1]

    def test_fresh_run_uses_the_latest_snapshot_lineage(self) -> None:
        """Events from an unrelated fresh run cannot count as current training."""
        events = [
            trained(1, rollout_id=0, indices=[1], lineage_id="old"),
            snapshot(2, rollout_id=0, holdings={}, lineage_id="old"),
            trained(3, rollout_id=0, indices=[1], lineage_id="new"),
            snapshot(4, rollout_id=0, holdings={}, lineage_id="new"),
        ]

        assert check(events) == []


class TestPolicyIsolation:
    @pytest.mark.parametrize("peer_fate", ["held", "dropped"])
    def test_a_peer_with_the_same_index_cannot_hide_policy_loss(self, peer_fate: str) -> None:
        """Another policy holding or dropping an index cannot account for a lost sample."""
        events = [
            generated(1, 1),
            snapshot(3, rollout_id=0, holdings={}, reason="save", trainer_model_id="solver"),
            snapshot(
                3,
                rollout_id=0,
                holdings={SampleOwner.OUTPUT_BUFFER: [1]} if peer_fate == "held" else {},
                reason="save",
                trainer_model_id="verifier",
            ),
        ]
        if peer_fate == "dropped":
            events.append(
                transition(2, indices=[1], to_owner=SampleOwner.DROPPED).model_copy(
                    update={"trainer_model_id": "verifier"}
                )
            )

        [issue] = [issue for partition in _partition_by_model_id(events) for issue in check(partition)]
        assert isinstance(issue, SampleLostIssue)
        assert issue.sample_indices == [1]

    def test_a_replay_policy_does_not_disable_fifo_duplicate_detection(self) -> None:
        """Replay declarations apply only to the policy whose buffer makes them."""
        events = [
            trained(1, rollout_id=0, indices=[1], trainer_model_id="solver"),
            trained(2, rollout_id=1, indices=[1], trainer_model_id="solver"),
            trained(1, rollout_id=0, indices=[1], trainer_model_id="verifier"),
            trained(2, rollout_id=1, indices=[1], trainer_model_id="verifier"),
            snapshot(3, rollout_id=1, holdings={}, trainer_model_id="solver"),
            snapshot(3, rollout_id=1, holdings={}, trainer_model_id="verifier", replays_samples=True),
        ]

        [issue] = [issue for partition in _partition_by_model_id(events) for issue in check(partition)]
        assert isinstance(issue, SampleConsumedTwiceIssue)
        assert issue.sample_indices == [1]


def restore(
    seconds: int,
    *,
    rollout_id: int,
    lineage_id: str,
    parent_lineage_id: str,
    rollout_ids: dict[str, int] | None = None,
) -> RolloutStateRestoreEvent:
    return RolloutStateRestoreEvent(
        timestamp=at(seconds),
        source=SOURCE,
        rollout_id=rollout_id,
        lineage_id=lineage_id,
        parent_lineage_id=parent_lineage_id,
        rollout_ids=rollout_ids,
    )


def _rank_events(
    seconds: int,
    *,
    rank: int,
    records: list[tuple[int, int]],
    cell: int = 0,
    role: Literal["actor", "critic"] = "actor",
    policy: str | None = None,
    rollout_id: int = 0,
    reason: Literal["step", "train_end", "save", "transfer", "load"] = "train_end",
    checkpoint_id: str | None = None,
) -> list:
    source = TrainProcessIdentity(component=role, model_id=policy, cell_index=cell, rank_within_cell=rank)
    mappings = [
        TrainerGroupMappingEvent(
            timestamp=at(seconds),
            source=source,
            lineage_id=None,
            trainer_model_id=policy,
            rollout_id=step,
            step_id=0,
            attempt=0,
            slot=None,
            groups={group: [group]},
        )
        for step, group in records
    ]
    return [
        *mappings,
        TrainerCpuWitnessEvent(
            timestamp=at(seconds),
            source=source,
            lineage_id=None,
            trainer_model_id=policy,
            rollout_id=rollout_id,
            records=[
                dict(
                    lineage_id=None,
                    trainer_model_id=policy,
                    rollout_id=step,
                    step_id=0,
                    attempt=0,
                    slot=None,
                    group_indices=[group],
                )
                for step, group in records
            ],
            reason=reason,
            checkpoint_id=checkpoint_id,
        ),
    ]


def _completion(
    seconds: int,
    *,
    rollout_id: int = 0,
    ranks: int = 2,
    cells: tuple[int, ...] = (0,),
    role: Literal["actor", "critic"] = "actor",
) -> TrainGroupStepEndEvent:
    return TrainGroupStepEndEvent(
        timestamp=at(seconds),
        source=SOURCE,
        rollout_id=rollout_id,
        role=role,
        cell_outcomes={cell: [TrainStepOutcome.NORMAL] * ranks for cell in cells},
    )


def _checkpoint(
    seconds: int, *, checkpoint_id: str, ranks: int = 2, cells: list[int] | None = None
) -> TrainerCheckpointEvent:
    return TrainerCheckpointEvent(
        timestamp=at(seconds),
        source=SOURCE,
        rollout_id=0,
        role="actor",
        checkpoint_id=checkpoint_id,
        cell_index=0,
        rank_count=ranks,
        alive_cell_indices=cells or [0],
    )


class TestRankWeightHistory:
    @pytest.mark.parametrize("peer_state", ["missing", "rolled_back", "retained"])
    def test_each_checkpoint_requires_fresh_peer_cell_evidence_after_same_rollout_reload(
        self, peer_state: str
    ) -> None:
        """A peer's old train completion cannot certify its weights after a same-rollout reload."""
        events = [
            generated(0, 7),
            *_rank_events(100, rank=0, records=[(0, 7)], reason="save", checkpoint_id="old"),
            *_rank_events(110, rank=0, cell=1, records=[(0, 7)], reason="save", checkpoint_id="old"),
            _checkpoint(2, checkpoint_id="old", ranks=1, cells=[0, 1]),
            snapshot(3, rollout_id=0, holdings={}, reason="save").model_copy(
                update={"checkpoint_ids": {"actor": "old"}}
            ),
            *_rank_events(120, rank=0, cell=1, records=[(0, 7)]),
            *_rank_events(130, rank=0, cell=1, records=[], reason="load"),
            *_rank_events(140, rank=0, records=[(0, 7)], reason="save", checkpoint_id="new"),
        ]
        if peer_state != "missing":
            events.extend(
                _rank_events(
                    150,
                    rank=0,
                    cell=1,
                    records=[(0, 7)] if peer_state == "retained" else [],
                    reason="save",
                    checkpoint_id="new",
                )
            )

        assert _check(events) == []
        events.extend(
            [
                _checkpoint(5, checkpoint_id="new", ranks=1, cells=[0, 1]),
                snapshot(6, rollout_id=0, holdings={}, reason="save").model_copy(
                    update={"checkpoint_ids": {"actor": "new"}}
                ),
            ]
        )
        issues = _check(events)
        assert bool(issues) == (peer_state != "retained")
        if issues:
            assert any(isinstance(issue, TrainerWeightHistoryIssue) for issue in issues)
            assert any(isinstance(issue, SampleLostIssue) and issue.sample_indices == [7] for issue in issues)
            assert all(issue.rollout_id == 0 for issue in issues)

    @pytest.mark.parametrize("previous_witness", [False, True])
    def test_rank_missing_across_advancing_rollouts_exhausts_runtime_tolerance(self, previous_witness: bool) -> None:
        """A rank that never appears or stops advancing remains the same missing rank across steps."""
        events = _rank_events(0, rank=1, records=[], rollout_id=0) if previous_witness else []
        for step in (1, 2, 3):
            events.extend(_rank_events(step * 3, rank=0, records=[], rollout_id=step))
            events.extend(
                [
                    _completion(step * 3 + 1, rollout_id=step),
                    snapshot(step * 3 + 2, rollout_id=step, holdings={}),
                ]
            )
        issues = _check(events)
        assert len(issues) == 1
        assert isinstance(issues[0], TrainerWeightHistoryIssue)
        assert issues[0].rollout_id == 3

    def test_advancing_lagging_rank_does_not_accumulate_permanent_missing_rank_failure(self) -> None:
        """A rank that advances one step behind and then catches up has separate transient gaps."""
        events = []
        for step in (1, 2, 3):
            events.extend(_rank_events(step * 3, rank=0, records=[], rollout_id=step))
            events.extend(_rank_events(step * 3, rank=1, records=[], rollout_id=step - 1))
            events.extend(
                [
                    _completion(step * 3 + 1, rollout_id=step),
                    snapshot(step * 3 + 2, rollout_id=step, holdings={}),
                ]
            )
        events.extend(_rank_events(12, rank=1, records=[], rollout_id=3))
        events.append(snapshot(13, rollout_id=3, holdings={}, reason="final"))
        assert _check(events) == []

    def test_new_active_cell_requires_its_restored_rank_evidence(self) -> None:
        """A healed cell cannot be invisible merely because it missed the previous training RPC."""
        events = [
            *_rank_events(1, rank=0, records=[(0, 7)]),
            _completion(2, ranks=1),
            CellReconfigureEvent(
                timestamp=at(3),
                source=SOURCE,
                rollout_id=0,
                quorum_id=1,
                src_cell_index=0,
                healed_cell_indices=[1],
                alive_cell_indices_after=[0, 1],
                role="actor",
            ),
            snapshot(4, rollout_id=0, holdings={}, reason="final"),
        ]
        assert any(isinstance(issue, TrainerWeightHistoryIssue) for issue in _check(events))

    def test_one_healthy_rank_cannot_hide_another_ranks_rollback(self) -> None:
        """Every rank must contain the saved training history even when a later peer looks healthy."""
        events = [
            generated(0, 7),
            *_rank_events(1, rank=0, records=[], reason="save", checkpoint_id="s"),
            *_rank_events(2, rank=1, records=[(0, 7)], reason="save", checkpoint_id="s"),
            _checkpoint(3, checkpoint_id="s"),
            snapshot(4, rollout_id=0, holdings={}, reason="save").model_copy(
                update={"checkpoint_ids": {"actor": "s"}}
            ),
        ]
        issues = _check(events)
        assert any(isinstance(issue, TrainerWeightHistoryIssue) for issue in issues)
        assert any(isinstance(issue, SampleLostIssue) and issue.sample_indices == [7] for issue in issues)

    def test_an_active_rank_that_never_emits_is_detected(self) -> None:
        """Controller rank results reveal a missing rank even if no rank log file exists."""
        events = [
            *_rank_events(1, rank=0, records=[(0, 7)]),
            _completion(2),
            snapshot(3, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: [7]}, reason="final"),
        ]
        assert any(isinstance(issue, TrainerWeightHistoryIssue) for issue in _check(events))

    def test_advancing_peer_does_not_compare_unfinished_rollout_to_completed_history(self) -> None:
        """A fast rank's next rollout is outside the previous successful controller boundary."""
        events = [
            *_rank_events(1, rank=0, records=[(0, 7), (1, 8)], rollout_id=1),
            *_rank_events(2, rank=1, records=[(0, 7)]),
            _completion(3),
            snapshot(4, rollout_id=1, holdings={SampleOwner.OUTPUT_BUFFER: [8]}, reason="final"),
        ]
        assert _check(events) == []

    def test_runtime_lag_is_evaluated_at_each_snapshot_and_can_recover(self) -> None:
        """Three old healthy snapshots cannot make one later delayed rank immediately fail."""
        events = [
            *_rank_events(1, rank=0, records=[(0, 7)]),
            *_rank_events(1, rank=1, records=[(0, 7)]),
            _completion(2),
        ]
        events.extend(snapshot(second, rollout_id=0, holdings={}) for second in (3, 4, 5))
        events.extend(
            [
                *_rank_events(6, rank=0, records=[]),
                snapshot(7, rollout_id=0, holdings={}),
                *_rank_events(8, rank=0, records=[(0, 7)]),
                snapshot(9, rollout_id=0, holdings={}),
            ]
        )
        assert _check(events) == []

    def test_persistent_rank_rollback_is_reported_after_three_real_defective_snapshots(self) -> None:
        """A persistent disagreement survives the runtime tolerance even if rollout retains the samples."""
        events = [*_rank_events(1, rank=0, records=[]), *_rank_events(1, rank=1, records=[(0, 7)]), _completion(2)]
        events.extend(
            snapshot(second, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: [7]}) for second in (3, 4, 5)
        )
        assert any(isinstance(issue, TrainerWeightHistoryIssue) for issue in _check(events))

    @pytest.mark.parametrize("conflict", [False, True])
    def test_restored_rank_can_use_donor_mapping_only_when_sources_agree(self, conflict: bool) -> None:
        """A donor mapping supports restored history, but an alternate source cannot overwrite it."""
        donor = _rank_events(1, rank=0, cell=1, records=[(0, 7)])
        restored = _rank_events(2, rank=0, records=[(0, 7)])[-1]
        events = [donor[0], restored, _completion(3, ranks=1), snapshot(4, rollout_id=0, holdings={}, reason="final")]
        if conflict:
            events.insert(1, donor[0].model_copy(update={"groups": {7: [99]}, "source": restored.source}))
        issues = _check(events)
        assert bool(issues) == conflict
        if conflict:
            assert isinstance(issues[0], TrainerWeightHistoryIssue)

    def test_dead_cell_is_excluded_from_completed_rank_comparison(self) -> None:
        """Membership retirement removes dead cells without requiring their final witness."""
        events = [
            *_rank_events(1, rank=0, records=[(0, 7)]),
            _completion(2, ranks=1, cells=(0, 1)),
            CellReconfigureEvent(
                timestamp=at(3),
                source=SOURCE,
                rollout_id=0,
                quorum_id=1,
                src_cell_index=0,
                healed_cell_indices=[],
                alive_cell_indices_after=[0],
                role="actor",
            ),
            snapshot(4, rollout_id=0, holdings={}, reason="final"),
        ]
        assert _check(events) == []

    def test_actor_and_critic_mappings_are_isolated_without_double_counting_training(self) -> None:
        """Actor and critic rank histories are checked separately and shared training counts once."""
        events = [
            *_rank_events(1, rank=0, records=[(0, 7)]),
            *_rank_events(1, rank=0, records=[(0, 8)], role="critic"),
            _completion(2, ranks=1),
            _completion(2, ranks=1, role="critic"),
            snapshot(3, rollout_id=0, holdings={}, reason="final"),
        ]
        assert _check(events) == []

    @pytest.mark.parametrize("replay", [False, True])
    def test_replica_agreement_does_not_hide_repeated_training_or_count_replicas_twice(self, replay: bool) -> None:
        """Replica histories are deduplicated while repeated optimizer consumption obeys replay policy."""
        events = [
            event for rank in (0, 1) for event in _rank_events(1, rank=rank, records=[(0, 7), (1, 7)], rollout_id=1)
        ]
        events.extend(
            [
                _completion(2, rollout_id=1),
                snapshot(3, rollout_id=1, holdings={}, reason="final", replays_samples=replay),
            ]
        )
        issues = _check(events)
        assert len(issues) == int(not replay)
        if issues:
            assert isinstance(issues[0], SampleConsumedTwiceIssue)

    def test_checkpoint_id_excludes_old_save_and_internal_transfer_even_with_clock_skew(self) -> None:
        """Only ranks bearing the current controller save identity prove the persisted weights."""
        events = [
            *_rank_events(100, rank=0, records=[(0, 7)], reason="save", checkpoint_id="old"),
            *_rank_events(101, rank=1, records=[(0, 7)], reason="save", checkpoint_id="old"),
            *_rank_events(102, rank=0, records=[], reason="save", checkpoint_id="new"),
            *_rank_events(103, rank=1, records=[], reason="transfer", checkpoint_id="new"),
            _checkpoint(2, checkpoint_id="new"),
            snapshot(3, rollout_id=0, holdings={}, reason="save").model_copy(
                update={"checkpoint_ids": {"actor": "new"}}
            ),
        ]
        assert any(isinstance(issue, TrainerWeightHistoryIssue) for issue in _check(events))

    def test_checkpoint_uses_all_live_cells_and_ignores_later_steps(self) -> None:
        """A saved cell and its live peer align at the saved step despite controller and rank clock skew."""
        events = [
            *_rank_events(100, rank=0, records=[(0, 7)], reason="save", checkpoint_id="s"),
            *_rank_events(110, rank=0, cell=1, records=[(0, 7)], reason="save", checkpoint_id="s"),
            *_rank_events(120, rank=0, cell=1, records=[(0, 7), (1, 8)], rollout_id=1),
            _checkpoint(200, checkpoint_id="s", ranks=1, cells=[0, 1]),
            snapshot(3, rollout_id=0, holdings={}, reason="save").model_copy(
                update={"checkpoint_ids": {"actor": "s"}}
            ),
        ]
        assert _check(events) == []

    def test_a_missing_unsaved_live_cell_is_detected_despite_controller_clock_skew(self) -> None:
        """The save completion names live cells even when controller timestamps exceed the snapshot."""
        events = [
            *_rank_events(100, rank=0, records=[(0, 7)], reason="save", checkpoint_id="s"),
            _checkpoint(200, checkpoint_id="s", ranks=1, cells=[0, 1]),
            snapshot(3, rollout_id=0, holdings={}, reason="save").model_copy(
                update={"checkpoint_ids": {"actor": "s"}}
            ),
        ]
        assert any(isinstance(issue, TrainerWeightHistoryIssue) for issue in _check(events))

    def test_later_load_does_not_replace_the_selected_saved_weight_incarnation(self) -> None:
        """A selected disk save remains the proof for its snapshot after subsequent rank resets."""
        events = [
            *_rank_events(100, rank=0, records=[(0, 7)], reason="save", checkpoint_id="s"),
            *_rank_events(110, rank=0, records=[], reason="load"),
            _checkpoint(200, checkpoint_id="s", ranks=1),
            generated(0, 7),
            snapshot(3, rollout_id=0, holdings={}, reason="save").model_copy(
                update={"checkpoint_ids": {"actor": "s"}}
            ),
        ]
        assert _check(events) == []

    def test_successive_steps_with_different_delayed_records_do_not_accumulate_one_defect(self) -> None:
        """Each changing delayed training record gets its own runtime tolerance window."""
        events = []
        for step in range(3):
            records = [(previous, previous + 7) for previous in range(step + 1)]
            events.extend(_rank_events(step * 4, rank=0, records=records, rollout_id=step))
            events.extend(_rank_events(step * 4, rank=1, records=records[:-1], rollout_id=step))
            events.extend(
                [_completion(step * 4 + 1, rollout_id=step), snapshot(step * 4 + 2, rollout_id=step, holdings={})]
            )
        assert _check(events) == []

    def test_a_peer_policy_cannot_supply_missing_rank_or_mapping_evidence(self) -> None:
        """Policy identity scopes both expected ranks and group mapping candidates."""
        events = [
            *_rank_events(1, rank=0, records=[(0, 7)], policy="a"),
            *_rank_events(2, rank=1, records=[(0, 7)], policy="b"),
            _completion(3),
            snapshot(4, rollout_id=0, holdings={}, reason="final", trainer_model_id="a"),
        ]
        assert any(isinstance(issue, TrainerWeightHistoryIssue) for issue in _check(events))

    def test_critic_warmup_training_is_counted_before_actor_training_begins(self) -> None:
        """Only critic warmup contributes separate consumption before actor training starts."""
        events = [
            generated(0, 7, 8),
            *_rank_events(1, rank=0, records=[(1, 8)], rollout_id=1),
            *_rank_events(1, rank=0, records=[(0, 7), (1, 8)], rollout_id=1, role="critic"),
            _completion(2, ranks=1, rollout_id=1),
            _completion(2, ranks=1, rollout_id=1, role="critic"),
            snapshot(3, rollout_id=1, holdings={}, reason="final"),
        ]
        assert _check(events) == []

    def test_online_mode_uses_only_three_snapshots_but_keeps_their_distinct_evidence(self) -> None:
        """Online analysis bounds snapshot work while preserving the same latest tolerance decision."""
        events = [
            *_rank_events(1, rank=0, records=[(0, 7)]),
            *_rank_events(1, rank=1, records=[(0, 7)]),
            _completion(2),
        ]
        events.extend(snapshot(second, rollout_id=0, holdings={}) for second in (3, 4, 5))
        events.extend(_rank_events(6, rank=0, records=[]))
        events.extend(snapshot(second, rollout_id=0, holdings={}) for second in (7, 8, 9))
        assert _check(events, latest_only=True) == _check(events)
