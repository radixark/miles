import pytest
from tests.fast.utils.soak.soak_fakes import _applied, _at, _cell_target, _observation, _request, _requested, _step_end
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.ft.checkers.trainer_peer_progress import assert_trainer_peers_progress
from tests.utils.soak.ft.types import CellTarget

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import TrainGroupStepEndEvent
from miles.utils.workers.naming import compute_cell_id

_VICTIM = _cell_target(kind="actor", cell_index=0, incarnation="victim-inc")
_PEER = _cell_target(kind="actor", cell_index=1, incarnation="peer-inc")


def _fault_on_victim(*, observed: list[CellTarget]) -> list[SoakEvent]:
    request = _request(_VICTIM, request_id="req-1")
    return [
        _observation(observed, at=_at(0)),
        _requested(request, at=_at(1)),
        _applied(request, at=_at(5)),
    ]


def _peer_step(
    *,
    second: float,
    incarnation: str = "peer-inc",
    outcomes: list[TrainStepOutcome] | str | None = None,
    cell_index: int = 1,
) -> TrainGroupStepEndEvent:
    step = _step_end(0, at=_at(second))
    return step.model_copy(
        update={
            "cell_outcomes": {cell_index: [TrainStepOutcome.NORMAL] if outcomes is None else outcomes},
            "cell_incarnations": {_PEER.identity: incarnation},
        }
    )


class TestAssertTrainerPeersProgress:
    def test_an_original_peer_completing_a_normal_step_after_the_fault_passes(self) -> None:
        """The healthy peer observed before the request trains on after the victim is hit."""
        assert_trainer_peers_progress(
            _fault_on_victim(observed=[_VICTIM, _PEER]), training_events=[_peer_step(second=6)]
        )

    def test_a_step_before_the_fault_effect_does_not_count(self) -> None:
        """Progress from before the effect says nothing about surviving it."""
        with pytest.raises(AssertionError, match="No original peer completed normal training"):
            assert_trainer_peers_progress(
                _fault_on_victim(observed=[_VICTIM, _PEER]), training_events=[_peer_step(second=4)]
            )

    def test_a_replaced_peer_does_not_count_as_the_original(self) -> None:
        """A peer restarted meanwhile is not the process that had to survive the fault."""
        with pytest.raises(AssertionError, match="No original peer completed normal training"):
            assert_trainer_peers_progress(
                _fault_on_victim(observed=[_VICTIM, _PEER]),
                training_events=[_peer_step(second=6, incarnation="peer-inc-2")],
            )

    @pytest.mark.parametrize(
        "outcomes",
        ["error", [], [TrainStepOutcome.NORMAL, TrainStepOutcome.DISCARDED_SHOULD_RETRY]],
        ids=["error", "empty", "discarded"],
    )
    def test_a_step_that_is_not_fully_normal_for_the_peer_does_not_count(
        self, outcomes: list[TrainStepOutcome] | str
    ) -> None:
        """An errored, empty or partly discarded step is not completed training."""
        with pytest.raises(AssertionError, match="No original peer completed normal training"):
            assert_trainer_peers_progress(
                _fault_on_victim(observed=[_VICTIM, _PEER]), training_events=[_peer_step(second=6, outcomes=outcomes)]
            )

    def test_the_outcome_is_read_from_the_peers_own_cell_index(self) -> None:
        """A normal outcome under another cell index belongs to another trainer."""
        with pytest.raises(AssertionError, match="No original peer completed normal training"):
            assert_trainer_peers_progress(
                _fault_on_victim(observed=[_VICTIM, _PEER]), training_events=[_peer_step(second=6, cell_index=0)]
            )

    def test_the_victim_itself_is_not_a_peer(self) -> None:
        """Without another observed trainer there is no survivor to check."""
        with pytest.raises(AssertionError, match="No original healthy peer was observed"):
            assert_trainer_peers_progress(_fault_on_victim(observed=[_VICTIM]), training_events=[])

    def test_a_trainer_of_another_pool_is_not_a_peer(self) -> None:
        """A different pool trains another model, so its progress is not the victim's group surviving."""
        other_pool = CellTarget(
            kind="actor",
            identity=compute_cell_id(pool_id="critic", cell_index=1),
            incarnation="peer-inc",
            alive=True,
            ready=True,
        )

        with pytest.raises(AssertionError, match="No original healthy peer was observed"):
            assert_trainer_peers_progress(_fault_on_victim(observed=[_VICTIM, other_pool]), training_events=[])

    def test_a_dead_trainer_is_not_a_peer(self) -> None:
        """Only trainers observed alive before the request are candidates."""
        dead = _PEER.model_copy(update={"alive": False})

        with pytest.raises(AssertionError, match="No original healthy peer was observed"):
            assert_trainer_peers_progress(_fault_on_victim(observed=[_VICTIM, dead]), training_events=[])

    def test_a_soak_without_an_applied_trainer_fault_fails(self) -> None:
        """A hook trigger that never hit a trainer must not pass vacuously."""
        request = _request(_VICTIM, request_id="req-1")

        with pytest.raises(AssertionError, match="No applied trainer fault had survivor evidence"):
            assert_trainer_peers_progress(
                [_observation([_VICTIM, _PEER], at=_at(0)), _requested(request, at=_at(1))],
                training_events=[_peer_step(second=6)],
            )
