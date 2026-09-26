import pytest
from tests.fast.utils.soak.soak_fakes import (
    _applied,
    _at,
    _cell_target,
    _observation,
    _reconfigure,
    _request,
    _requested,
)
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import SoakActionRecord, project_actions
from tests.utils.soak.ft.recovery import compute_recovered_at
from tests.utils.soak.ft.types import CellTarget


def _action(target: CellTarget, *, applied: bool = True) -> tuple[SoakActionRecord, list[SoakEvent]]:
    request = _request(target, request_id=f"req-{target.identity}")
    events: list[SoakEvent] = [_requested(request, at=_at(0))]
    if applied:
        events.append(_applied(request, at=_at(1)))
    return project_actions(events)[request.request_id], events


def _recovered_at(target: CellTarget, later: list[SoakEvent], *, applied: bool = True) -> object:
    action, events = _action(target, applied=applied)
    return compute_recovered_at(action=action, events=[*events, *later])


class TestRolloutRecovery:
    def test_a_ready_new_incarnation_after_the_effect_is_recovery(self) -> None:
        """A rollout cell recovers when a fresh incarnation of the same cell is observed ready."""
        target = _cell_target(kind="rollout", cell_index=1)
        fresh = target.model_copy(update={"incarnation": "inc-b"})

        assert _recovered_at(target, [_observation([fresh], at=_at(5))]) == _at(5)

    def test_an_action_without_an_effect_never_recovers(self) -> None:
        """Nothing can recover from a fault that never landed."""
        target = _cell_target(kind="rollout", cell_index=1)
        fresh = target.model_copy(update={"incarnation": "inc-b"})

        assert _recovered_at(target, [_observation([fresh], at=_at(5))], applied=False) is None

    def test_the_old_incarnation_reading_ready_is_not_recovery(self) -> None:
        """A stale ready reading of the killed incarnation proves nothing."""
        target = _cell_target(kind="rollout", cell_index=1)

        assert _recovered_at(target, [_observation([target], at=_at(5))]) is None

    def test_a_new_incarnation_counts_only_once_it_is_ready(self) -> None:
        """A replacement still starting up has not recovered yet."""
        target = _cell_target(kind="rollout", cell_index=1)
        starting = target.model_copy(update={"incarnation": "inc-b", "ready": False})
        serving = target.model_copy(update={"incarnation": "inc-b"})

        assert _recovered_at(target, [_observation([starting], at=_at(5))]) is None
        assert _recovered_at(target, [_observation([starting], at=_at(5)), _observation([serving], at=_at(7))]) == _at(
            7
        )

    def test_an_observation_from_before_the_effect_is_ignored(self) -> None:
        """A new incarnation seen before the fault landed is not the fault's recovery."""
        target = _cell_target(kind="rollout", cell_index=1)
        fresh = target.model_copy(update={"incarnation": "inc-b"})
        action, events = _action(target)

        assert compute_recovered_at(action=action, events=[_observation([fresh], at=_at(0.5)), *events]) is None

    def test_a_siblings_new_incarnation_does_not_recover_the_target(self) -> None:
        """Only the injected cell's own replacement counts."""
        target = _cell_target(kind="rollout", cell_index=1)
        sibling = _cell_target(kind="rollout", cell_index=2, incarnation="inc-b")

        assert _recovered_at(target, [_observation([sibling], at=_at(5))]) is None

    def test_an_empty_incarnation_cannot_witness_recovery(self) -> None:
        """Without a known incarnation there is nothing to compare the replacement against."""
        target = _cell_target(kind="rollout", cell_index=1, incarnation="")

        with pytest.raises(AssertionError, match="nonempty target incarnation"):
            _recovered_at(target, [])


class TestTrainerRecovery:
    def _fresh(self) -> tuple[CellTarget, CellTarget]:
        target = _cell_target(cell_index=1)
        return target, target.model_copy(update={"incarnation": "inc-b"})

    def test_a_ready_new_incarnation_healed_into_the_group_is_recovery(self) -> None:
        """A trainer recovers once a reconfigure readmitted its index with the new incarnation."""
        target, fresh = self._fresh()
        healing = _reconfigure(at=_at(3), healed_cell_indices=[1], cell_incarnations_after={target.identity: "inc-b"})

        assert _recovered_at(target, [_observation([fresh], at=_at(5), new_sut_events=[healing])]) == _at(5)

    def test_a_ready_new_incarnation_without_a_reconfigure_is_not_recovery(self) -> None:
        """A restarted trainer outside the training group has not healed."""
        target, fresh = self._fresh()

        assert _recovered_at(target, [_observation([fresh], at=_at(5))]) is None

    @pytest.mark.parametrize(
        ("at", "healed_cell_indices", "incarnation"),
        [(-1, [1], "inc-b"), (6, [1], "inc-b"), (3, [2], "inc-b"), (3, [1], "inc-c")],
    )
    def test_a_reconfigure_outside_the_window_of_another_cell_or_incarnation_is_not_recovery(
        self, at: float, healed_cell_indices: list[int], incarnation: str
    ) -> None:
        """The reconfigure must follow the request, precede the observation and name this cell's new incarnation."""
        target, fresh = self._fresh()
        healing = _reconfigure(
            at=_at(at), healed_cell_indices=healed_cell_indices, cell_incarnations_after={target.identity: incarnation}
        )

        assert _recovered_at(target, [_observation([fresh], at=_at(5), new_sut_events=[healing])]) is None

    def test_a_later_observation_after_a_late_reconfigure_is_recovery(self) -> None:
        """Recovery is dated by the first observation that follows the matching reconfigure."""
        target, fresh = self._fresh()
        healing = _reconfigure(at=_at(6), healed_cell_indices=[1], cell_incarnations_after={target.identity: "inc-b"})

        assert _recovered_at(
            target,
            [_observation([fresh], at=_at(5), new_sut_events=[healing]), _observation([fresh], at=_at(7))],
        ) == _at(7)

    def test_one_reconfigure_healing_two_cells_recovers_both(self) -> None:
        """One reconfigure can readmit several cells, and each of them counts as healed."""
        first, second = _cell_target(cell_index=1), _cell_target(cell_index=2)
        healing = _reconfigure(
            at=_at(3),
            healed_cell_indices=[1, 2],
            cell_incarnations_after={first.identity: "inc-b", second.identity: "inc-c"},
        )
        observed = _observation(
            [first.model_copy(update={"incarnation": "inc-b"}), second.model_copy(update={"incarnation": "inc-c"})],
            at=_at(5),
            new_sut_events=[healing],
        )

        assert _recovered_at(first, [observed]) == _at(5)
        assert _recovered_at(second, [observed]) == _at(5)
