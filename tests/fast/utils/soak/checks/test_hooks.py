from datetime import timedelta

import pytest
from tests.fast.utils.soak.checks.conftest import HookEvidence
from tests.fast.utils.soak.utils import typed_cell
from tests.utils.soak.checks.hooks import (
    assert_batch_trainers_recovered,
    assert_hook_effects,
    assert_hook_survivors,
    assert_remote_p2p_failures,
)
from tests.utils.soak.state import SoakActionAppliedEvent, SoakActionRequest, SoakActionRequestedEvent, SoakObservation

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import TrainGroupStepEndEvent, WeightUpdateResultEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity


class TestHookEffects:
    @pytest.mark.parametrize("recorded_delay", [500.0, 700.0])
    def test_random_delay_must_match_the_recorded_draw(
        self, hook_evidence: HookEvidence, recorded_delay: float
    ) -> None:
        """A valid hook hit with a different delay cannot satisfy the persisted random request."""
        original = hook_evidence.events[0].request
        request = original.model_copy(
            update={"form_name": "hook:trainer_before_all_gather:sigkill:random", "hook_delay_ms": recorded_delay}
        )
        events = [SoakActionRequestedEvent(request=request), *hook_evidence.events[1:]]
        if recorded_delay == 500:
            assert_hook_effects(events, hook_events=[hook_evidence.hit])
        else:
            with pytest.raises(AssertionError, match="recorded draw"):
                assert_hook_effects(events, hook_events=[hook_evidence.hit])

    def test_matching_dispatch_and_effect_are_accepted(self, hook_evidence: HookEvidence) -> None:
        """A unique dispatch with the requested delay proves a precise effect."""
        assert_hook_effects(hook_evidence.events, hook_events=[hook_evidence.hit])

    @pytest.mark.parametrize("count", [0, 2])
    def test_missing_or_duplicate_dispatch_is_rejected(self, hook_evidence: HookEvidence, count: int) -> None:
        """An effect receipt alone or duplicate dispatches cannot prove one precise injection."""
        with pytest.raises(AssertionError):
            assert_hook_effects(hook_evidence.events, hook_events=[hook_evidence.hit] * count)

    @pytest.mark.parametrize(
        "updates",
        [
            {"instance_id": "replacement"},
            {"hook": "trainer_before_weight_send"},
            {"mode": "exit"},
            {"weight_version": None},
            {"update_id": None},
            {"reached_at": None},
            {"due_at": 11.1},
            {"monotonic_time": 11.4},
        ],
    )
    def test_wrong_identity_or_timing_is_rejected(self, hook_evidence: HookEvidence, updates: dict) -> None:
        """Unrelated incarnations and early or incorrectly delayed faults do not count as hits."""
        with pytest.raises(AssertionError):
            assert_hook_effects(hook_evidence.events, hook_events=[hook_evidence.hit.model_copy(update=updates)])

    @pytest.mark.parametrize("status", ["cancelled", "expired", "failed"])
    def test_contradictory_terminal_evidence_is_rejected(self, hook_evidence: HookEvidence, status: str) -> None:
        """An applied action cannot also claim its dispatch was prevented or failed."""
        with pytest.raises(AssertionError, match="also claims"):
            assert_hook_effects(
                hook_evidence.events,
                hook_events=[hook_evidence.hit, hook_evidence.hit.model_copy(update={"status": status})],
            )

    def test_duplicate_effect_is_rejected(self, hook_evidence: HookEvidence) -> None:
        """Repeated receipts cannot inflate precise fault coverage."""
        with pytest.raises(AssertionError, match="Duplicate hook effect"):
            assert_hook_effects([*hook_evidence.events, hook_evidence.events[-1]], hook_events=[hook_evidence.hit])


class TestRemoteHookEffects:
    def test_matching_trigger_and_victim_evidence_are_accepted(self, remote_hook_evidence: HookEvidence) -> None:
        """Remote success requires both the original trainer hit and the independent victim receipt."""
        assert_hook_effects(remote_hook_evidence.events, hook_events=[remote_hook_evidence.hit])

    @pytest.mark.parametrize("corruption", ["trigger", "victim", "version", "mode", "empty_receipt"])
    def test_mixed_or_incomplete_remote_evidence_is_rejected(
        self, remote_hook_evidence: HookEvidence, corruption: str
    ) -> None:
        """A valid trainer hit cannot authenticate unrelated or incomplete victim evidence."""
        evidence = remote_hook_evidence.events[-1].evidence
        if corruption == "trigger":
            evidence["hook_trigger"]["workers_hash"] = "replacement"
        elif corruption == "victim":
            evidence["target"]["workers_hash"] = "replacement"
        elif corruption == "version":
            evidence["hook_hit"]["weight_version"] = 99
        elif corruption == "mode":
            evidence["mode"] = "exit"
        else:
            evidence["exited_pids"] = []

        with pytest.raises((AssertionError, ValueError)):
            assert_hook_effects(remote_hook_evidence.events, hook_events=[remote_hook_evidence.hit])


class TestRemoteP2PFailures:
    @pytest.mark.parametrize("different_form", [False, True])
    def test_late_miss_requires_a_later_precise_hit_from_the_same_form(
        self, remote_hook_evidence: HookEvidence, p2p_update_result: WeightUpdateResultEvent, different_form: bool
    ) -> None:
        """A real late fault stays recorded but cannot supply another form's precise-hit coverage."""
        original = remote_hook_evidence.events[0].request
        later = original.model_copy(update={"request_id": "later"})
        receipt = remote_hook_evidence.events[-1].model_copy(deep=True)
        receipt.request_id = later.request_id
        receipt.evidence["request_id"] = later.request_id
        receipt.evidence["hook_request"]["request_id"] = "later:trigger"
        receipt.evidence["hook_hit"]["request"]["request_id"] = "later:trigger"
        receipt.evidence["hook_hit"]["update_id"] = "later-update"
        hit = remote_hook_evidence.hit.model_copy(update={"request_id": "later:trigger", "update_id": "later-update"})
        if different_form:
            later = later.model_copy(update={"form_name": original.form_name.replace("500ms", "700ms")})
            receipt.evidence["hook_request"]["delay_ms"] = 700
            receipt.evidence["hook_hit"]["request"]["delay_ms"] = 700
            receipt.evidence["hook_hit"]["due_at"] = 11.7
            receipt.evidence["hook_hit"]["changed_at"] = 11.8
            hit = hit.model_copy(update={"due_at": 11.7, "monotonic_time": 11.8})
        events = [*remote_hook_evidence.events, SoakActionRequestedEvent(request=later), receipt]
        results = [
            p2p_update_result.model_copy(
                update={"updated_cell_ids": ["rollout-1", "rollout-2"], "failed_cell_ids": []}
            ),
            p2p_update_result.model_copy(update={"update_id": "later-update"}),
        ]
        if different_form:
            with pytest.raises(AssertionError, match="forms without a precise hit"):
                assert_remote_p2p_failures(events, hook_events=[remote_hook_evidence.hit, hit], update_events=results)
        else:
            assert assert_remote_p2p_failures(
                events, hook_events=[remote_hook_evidence.hit, hit], update_events=results
            ) == {later.request_id}

    @pytest.mark.parametrize("outcome", ["sender_failed", "partial", "no_peer", "wrong_assignment"])
    def test_sender_failure_requires_its_complete_assignment_and_a_publishing_peer(
        self, batch_remote_hook_evidence: HookEvidence, p2p_update_result: WeightUpdateResultEvent, outcome: str
    ) -> None:
        """A sender's failed partition must leave another sender able to publish and restore it."""
        updates = {
            "target_incarnations": {f"rollout-{index}": "generation-0" for index in range(1, 4)},
            "updated_cell_ids": ["rollout-3"],
            "failed_cell_ids": ["rollout-1", "rollout-2"],
            "published_version": 23,
        }
        if outcome == "partial":
            updates.update(updated_cell_ids=["rollout-2", "rollout-3"], failed_cell_ids=["rollout-1"])
        elif outcome == "no_peer":
            updates.update(
                updated_cell_ids=[], failed_cell_ids=["rollout-1", "rollout-2", "rollout-3"], published_version=None
            )
        elif outcome == "wrong_assignment":
            batch_remote_hook_evidence.hit.target_incarnations["rollout-3"] = "generation-0"
        result = p2p_update_result.model_copy(update=updates)

        if outcome == "sender_failed":
            assert_remote_p2p_failures(
                batch_remote_hook_evidence.events,
                hook_events=[batch_remote_hook_evidence.hit],
                update_events=[result],
                require_all_targets_failed=True,
            )
        else:
            with pytest.raises(AssertionError):
                assert_remote_p2p_failures(
                    batch_remote_hook_evidence.events,
                    hook_events=[batch_remote_hook_evidence.hit],
                    update_events=[result],
                    require_all_targets_failed=True,
                )

    def test_partial_failure_is_bound_to_the_triggered_update(
        self, remote_hook_evidence: HookEvidence, p2p_update_result: WeightUpdateResultEvent
    ) -> None:
        """A failed victim and a surviving target in the same update prove partial failure."""
        assert_remote_p2p_failures(
            remote_hook_evidence.events,
            hook_events=[remote_hook_evidence.hit],
            update_events=[p2p_update_result],
        )

    @pytest.mark.parametrize(
        "updates",
        [
            {"update_id": "same-version-different-attempt"},
            {"candidate_version": 24},
            {"target_incarnations": {"rollout-1": "replacement", "rollout-2": "generation-0"}},
            {"updated_cell_ids": ["rollout-1", "rollout-2"], "failed_cell_ids": []},
            {"updated_cell_ids": ["rollout-1", "rollout-2"]},
            {"updated_cell_ids": []},
            {"published_version": None},
        ],
    )
    def test_other_attempts_and_inconsistent_results_cannot_prove_a_p2p_failure(
        self, remote_hook_evidence: HookEvidence, p2p_update_result: WeightUpdateResultEvent, updates: dict
    ) -> None:
        """A reused version, replaced victim or missed transfer cannot supply failure coverage."""
        with pytest.raises(AssertionError):
            assert_remote_p2p_failures(
                remote_hook_evidence.events,
                hook_events=[remote_hook_evidence.hit],
                update_events=[p2p_update_result.model_copy(update=updates)],
            )

    @pytest.mark.parametrize("count", [0, 2])
    def test_missing_or_duplicated_update_results_are_rejected(
        self, remote_hook_evidence: HookEvidence, p2p_update_result: WeightUpdateResultEvent, count: int
    ) -> None:
        """A trigger must join exactly one completed update result."""
        with pytest.raises(AssertionError):
            assert_remote_p2p_failures(
                remote_hook_evidence.events,
                hook_events=[remote_hook_evidence.hit],
                update_events=[p2p_update_result] * count,
            )


class TestBatchTrainerRecovery:
    @pytest.mark.parametrize(
        "case", ["recovered", "unchanged", "missing", "discarded", "other_trainer", "before", "peer_replaced"]
    )
    def test_retired_sender_resumes_training_with_its_original_peer(
        self, batch_request: SoakActionRequest, case: str
    ) -> None:
        """The targeted sender must recover while its original peer remains alive and trains normally."""
        requested = SoakActionRequestedEvent(request=batch_request)
        applied = SoakActionAppliedEvent(request_id=batch_request.request_id, evidence={})
        step = TrainGroupStepEndEvent(
            timestamp=applied.timestamp + timedelta(seconds=-1 if case == "before" else 1),
            source=TrainerControllerProcessIdentity(trainer_id="critic" if case == "other_trainer" else "actor"),
            rollout_id=24,
            cell_outcomes={
                0: [TrainStepOutcome.NORMAL],
                1: [TrainStepOutcome.DISCARDED_SHOULD_RETRY if case == "discarded" else TrainStepOutcome.NORMAL],
            },
            cell_incarnations={
                "actor-0": "generation-0" if case == "unchanged" else "new-0",
                "actor-1": "new-1" if case == "peer_replaced" else "generation-0",
            },
        )
        if case == "missing":
            step.cell_incarnations.pop("actor-1")
        events = [
            SoakObservation(cells=[typed_cell(f"actor-{index}", "actor") for index in range(2)]),
            requested,
            applied,
        ]
        if case == "recovered":
            assert_batch_trainers_recovered(
                events, steps=[step], expected_trainers=2, matched_request_ids={batch_request.request_id}
            )
        else:
            with pytest.raises(AssertionError):
                assert_batch_trainers_recovered(
                    events, steps=[step], expected_trainers=2, matched_request_ids={batch_request.request_id}
                )


class TestHookSurvivors:
    @pytest.mark.parametrize("case", ["survived", "replaced", "discarded", "empty", "before_fault", "no_peer"])
    def test_only_original_peers_with_new_normal_training_count(self, hook_evidence: HookEvidence, case: str) -> None:
        """Recreated peers and discarded or pre-fault attempts cannot prove surviving progress."""
        peer = typed_cell("actor-8", "actor")
        observation = SoakObservation(cells=[] if case == "no_peer" else [peer])
        step = TrainGroupStepEndEvent(
            timestamp=hook_evidence.events[-1].timestamp + timedelta(seconds=-1 if case == "before_fault" else 1),
            source=TrainerControllerProcessIdentity(trainer_id="actor"),
            rollout_id=24,
            cell_outcomes={
                8: (
                    []
                    if case == "empty"
                    else [TrainStepOutcome.DISCARDED_SHOULD_RETRY if case == "discarded" else TrainStepOutcome.NORMAL]
                )
            },
            cell_incarnations={"actor-8": "replacement" if case == "replaced" else peer["status"]["workers_hash"]},
        )

        if case == "survived":
            assert_hook_survivors([observation, *hook_evidence.events], steps=[step])
        else:
            with pytest.raises(AssertionError):
                assert_hook_survivors([observation, *hook_evidence.events], steps=[step])
