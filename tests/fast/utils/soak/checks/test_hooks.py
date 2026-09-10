from datetime import timedelta

import pytest
from tests.fast.utils.soak.checks.conftest import HookEvidence
from tests.fast.utils.soak.utils import typed_cell
from tests.utils.soak.checks.hooks import assert_hook_effects, assert_hook_survivors
from tests.utils.soak.state import SoakObservation

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity


class TestHookEffects:
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
