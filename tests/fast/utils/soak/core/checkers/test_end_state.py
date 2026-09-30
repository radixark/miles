import pytest
from tests.fast.utils.soak.soak_fakes import _at, _cell_target, _observation
from tests.utils.soak.core.checkers.end_state import assert_end_state_complete
from tests.utils.soak.core.events import SoakAdmissionClosedEvent

_EXPECTED = {"actor": 2, "rollout": 1}


def _complete() -> list:
    return [_cell_target(cell_index=0), _cell_target(cell_index=1), _cell_target(kind="rollout")]


def _closed(seconds: float) -> SoakAdmissionClosedEvent:
    return SoakAdmissionClosedEvent(timestamp=_at(seconds))


class TestAssertEndStateComplete:
    def test_a_final_observation_with_every_target_alive_and_ready_passes(self) -> None:
        """The run ends whole when each kind has its expected ready targets."""
        assert_end_state_complete(
            [_observation(_complete(), at=_at(0)), _closed(1)],
            expected_count_of_kind=_EXPECTED,
            observation_ends_with_sut=False,
        )

    def test_a_run_without_observations_is_rejected(self) -> None:
        """Nothing observed means nothing proven about the end state."""
        with pytest.raises(AssertionError, match="without any observation"):
            assert_end_state_complete([], expected_count_of_kind=_EXPECTED, observation_ends_with_sut=False)

    def test_a_failed_final_observation_is_rejected(self) -> None:
        """A final poll that saw no targets cannot show a whole run."""
        with pytest.raises(AssertionError, match="failed observation"):
            assert_end_state_complete(
                [_observation(_complete(), at=_at(0)), _observation(None, at=_at(1))],
                expected_count_of_kind=_EXPECTED,
                observation_ends_with_sut=False,
            )

    def test_a_final_observation_with_errors_is_rejected(self) -> None:
        """A partial final view is not proof of completeness."""
        with pytest.raises(AssertionError, match="with errors"):
            assert_end_state_complete(
                [_observation(_complete(), at=_at(0), errors={"pods": "down"})],
                expected_count_of_kind=_EXPECTED,
                observation_ends_with_sut=False,
            )

    @pytest.mark.parametrize(
        "targets",
        [
            [_cell_target(cell_index=0), _cell_target(kind="rollout")],
            [
                _cell_target(cell_index=0),
                _cell_target(cell_index=1),
                _cell_target(cell_index=2),
                _cell_target(kind="rollout"),
            ],
            [_cell_target(cell_index=0), _cell_target(cell_index=1)],
        ],
    )
    def test_a_kind_with_the_wrong_target_count_is_rejected(self, targets: list) -> None:
        """Missing or leftover targets of any kind fail the end state."""
        with pytest.raises(AssertionError, match="targets, expected"):
            assert_end_state_complete(
                [_observation(targets, at=_at(0))], expected_count_of_kind=_EXPECTED, observation_ends_with_sut=False
            )

    @pytest.mark.parametrize("state", [{"alive": False}, {"ready": False}])
    def test_a_dead_or_unready_target_is_rejected(self, state: dict) -> None:
        """Every target must be both alive and ready at the end."""
        targets = [_cell_target(cell_index=0), _cell_target(cell_index=1, **state), _cell_target(kind="rollout")]

        with pytest.raises(AssertionError, match="not alive and ready"):
            assert_end_state_complete(
                [_observation(targets, at=_at(0))], expected_count_of_kind=_EXPECTED, observation_ends_with_sut=False
            )

    def test_only_the_latest_observation_is_judged(self) -> None:
        """A healthy earlier poll cannot mask a broken final one."""
        broken = [_cell_target(cell_index=0), _cell_target(cell_index=1, ready=False), _cell_target(kind="rollout")]

        with pytest.raises(AssertionError, match="not alive and ready"):
            assert_end_state_complete(
                [_observation(_complete(), at=_at(0)), _observation(broken, at=_at(1))],
                expected_count_of_kind=_EXPECTED,
                observation_ends_with_sut=False,
            )


class TestAnObservationThatEndsWithTheSut:
    def test_failed_polls_after_the_training_exit_fall_back_to_the_last_successful_one(self) -> None:
        """A finished training run takes its api server down, so its final polls fail by design."""
        assert_end_state_complete(
            [
                _closed(0),
                _observation(_complete(), at=_at(1)),
                _observation(None, at=_at(2)),
                _observation(None, at=_at(3)),
            ],
            expected_count_of_kind=_EXPECTED,
            observation_ends_with_sut=True,
        )

    def test_a_last_successful_observation_before_the_last_fault_is_rejected(self) -> None:
        """A whole view from before the tail proves nothing about how the run ended."""
        with pytest.raises(AssertionError, match="before its last fault"):
            assert_end_state_complete(
                [_observation(_complete(), at=_at(0)), _closed(1), _observation(None, at=_at(2))],
                expected_count_of_kind=_EXPECTED,
                observation_ends_with_sut=True,
            )

    def test_a_trailing_observation_with_errors_is_not_skipped(self) -> None:
        """Only polls that saw nothing are attributed to the exit; a partial view still fails."""
        with pytest.raises(AssertionError, match="with errors"):
            assert_end_state_complete(
                [
                    _closed(0),
                    _observation(_complete(), at=_at(1)),
                    _observation(_complete(), at=_at(2), errors={"cells": "timeout"}),
                    _observation(None, at=_at(3)),
                ],
                expected_count_of_kind=_EXPECTED,
                observation_ends_with_sut=True,
            )

    def test_only_failed_polls_are_rejected(self) -> None:
        """A run whose every poll failed never showed its end state."""
        with pytest.raises(AssertionError, match="without any observation"):
            assert_end_state_complete(
                [_closed(0), _observation(None, at=_at(1))],
                expected_count_of_kind=_EXPECTED,
                observation_ends_with_sut=True,
            )
