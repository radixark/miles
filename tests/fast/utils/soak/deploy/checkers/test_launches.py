import pytest
from tests.fast.utils.soak.deploy.deploy_fakes import (
    _deployment_target,
    _hot_restart_request,
    _landed_take_over,
    _requested_take_over,
)
from tests.fast.utils.soak.soak_fakes import _applied, _at, _cell_target, _request, _requested
from tests.utils.soak.core.events import LaunchOutcome, SoakEvent, SoakLaunchFinishedEvent
from tests.utils.soak.deploy.checkers.launches import assert_hot_restart_launches_finished


def _take_over(request_id: str, *, requested_at: float, landed_at: float | None) -> list[SoakEvent]:
    target = _deployment_target()
    request = _hot_restart_request(target, request_id=request_id)
    events: list[SoakEvent] = [_requested_take_over(request, at=_at(requested_at))]
    if landed_at is not None:
        events.append(_landed_take_over(request, after=target, at=_at(landed_at)))
    return events


def _ended(request_id: str | None, outcome: LaunchOutcome, *, at: float) -> SoakLaunchFinishedEvent:
    return SoakLaunchFinishedEvent(timestamp=_at(at), request_id=request_id, outcome=outcome)


def _sorted(events: list[SoakEvent]) -> list[SoakEvent]:
    return sorted(events, key=lambda event: event.timestamp)


class TestAssertHotRestartLaunchesFinished:
    def test_a_chain_of_replaced_launchers_each_explained_by_the_next_take_over_passes(self) -> None:
        """Every REPLACED exit is followed by the take-over that replaced it, and the last one finishes."""
        events = _sorted(
            [
                *_take_over("a", requested_at=1, landed_at=3),
                _ended(None, LaunchOutcome.REPLACED, at=2),
                *_take_over("b", requested_at=5, landed_at=7),
                _ended("a", LaunchOutcome.REPLACED, at=6),
                _ended("b", LaunchOutcome.FINISHED, at=9),
            ]
        )

        assert_hot_restart_launches_finished(events)

    def test_a_replacement_exit_after_the_successor_landed_passes(self) -> None:
        """Landing may win the race against the old launcher noticing its SIGTERM."""
        events = _sorted([*_take_over("a", requested_at=1, landed_at=2), _ended(None, LaunchOutcome.REPLACED, at=4)])

        assert_hot_restart_launches_finished(events)

    @pytest.mark.parametrize("request_id", [None, "a"])
    def test_any_failed_launcher_fails(self, request_id: str | None) -> None:
        """A failed initial or take-over launcher is a failed run whatever else landed."""
        events = _sorted(
            [*_take_over("a", requested_at=1, landed_at=2), _ended(request_id, LaunchOutcome.FAILED, at=3)]
        )

        with pytest.raises(AssertionError, match="Launcher failed"):
            assert_hot_restart_launches_finished(events)

    def test_a_replacement_exit_before_the_successor_was_requested_fails(self) -> None:
        """A later take-over cannot explain an exit captured before that replacement started."""
        events = _sorted([_ended(None, LaunchOutcome.REPLACED, at=1), *_take_over("a", requested_at=2, landed_at=3)])

        with pytest.raises(AssertionError, match="No applied successor"):
            assert_hot_restart_launches_finished(events)

    def test_a_replacement_exit_whose_successor_never_landed_fails(self) -> None:
        """A started replacement needs its own landing to explain the SIGTERM exit."""
        events = _sorted(
            [*_take_over("a", requested_at=1, landed_at=None), _ended(None, LaunchOutcome.REPLACED, at=2)]
        )

        with pytest.raises(AssertionError, match="No applied successor"):
            assert_hot_restart_launches_finished(events)

    def test_only_the_exact_next_take_over_explains_a_replacement(self) -> None:
        """A landing two take-overs later does not explain the launcher its unlanded predecessor replaced."""
        events = _sorted(
            [
                *_take_over("a", requested_at=1, landed_at=None),
                _ended(None, LaunchOutcome.REPLACED, at=2),
                *_take_over("b", requested_at=3, landed_at=4),
            ]
        )

        with pytest.raises(AssertionError, match="No applied successor"):
            assert_hot_restart_launches_finished(events)

    def test_the_final_launcher_being_replaced_fails(self) -> None:
        """Nothing follows the last take-over, so its SIGTERM exit is an unexplained kill."""
        events = _sorted([*_take_over("a", requested_at=1, landed_at=2), _ended("a", LaunchOutcome.REPLACED, at=3)])

        with pytest.raises(AssertionError, match="final launcher was replaced"):
            assert_hot_restart_launches_finished(events)

    def test_cell_fault_actions_are_not_successors(self) -> None:
        """Only deployment take-overs replace a launcher, so a landed cell fault cannot explain one."""
        cell_request = _request(_cell_target(), request_id="cell")
        events = [
            _requested(cell_request, at=_at(1)),
            _applied(cell_request, at=_at(2)),
            _ended(None, LaunchOutcome.REPLACED, at=3),
        ]

        with pytest.raises(AssertionError, match="final launcher was replaced"):
            assert_hot_restart_launches_finished(events)
