import pytest
from tests.fast.utils.soak.soak_fakes import _at, _cell_target, _observation
from tests.utils.soak.core.checkers.end_state import assert_end_state_complete
from tests.utils.soak.core.events import SoakAdmissionClosedEvent

_EXPECTED = {"actor": 2, "rollout": 1}


def _complete() -> list:
    return [_cell_target(cell_index=0), _cell_target(cell_index=1), _cell_target(kind="rollout")]


class TestAssertEndStateComplete:
    def test_a_final_observation_with_every_target_alive_and_ready_passes(self) -> None:
        """The run ends whole when each kind has its expected ready targets."""
        assert_end_state_complete(
            [_observation(_complete(), at=_at(0)), SoakAdmissionClosedEvent(timestamp=_at(1))],
            expected_count_of_kind=_EXPECTED,
        )

    def test_a_run_without_observations_is_rejected(self) -> None:
        """Nothing observed means nothing proven about the end state."""
        with pytest.raises(AssertionError, match="without any observation"):
            assert_end_state_complete([], expected_count_of_kind=_EXPECTED)

    def test_a_failed_final_observation_is_rejected(self) -> None:
        """A final poll that saw no targets cannot show a whole run."""
        with pytest.raises(AssertionError, match="failed observation"):
            assert_end_state_complete(
                [_observation(_complete(), at=_at(0)), _observation(None, at=_at(1))], expected_count_of_kind=_EXPECTED
            )

    def test_a_final_observation_with_errors_is_rejected(self) -> None:
        """A partial final view is not proof of completeness."""
        with pytest.raises(AssertionError, match="with errors"):
            assert_end_state_complete(
                [_observation(_complete(), at=_at(0), errors={"pods": "down"})], expected_count_of_kind=_EXPECTED
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
            assert_end_state_complete([_observation(targets, at=_at(0))], expected_count_of_kind=_EXPECTED)

    @pytest.mark.parametrize("state", [{"alive": False}, {"ready": False}])
    def test_a_dead_or_unready_target_is_rejected(self, state: dict) -> None:
        """Every target must be both alive and ready at the end."""
        targets = [_cell_target(cell_index=0), _cell_target(cell_index=1, **state), _cell_target(kind="rollout")]

        with pytest.raises(AssertionError, match="not alive and ready"):
            assert_end_state_complete([_observation(targets, at=_at(0))], expected_count_of_kind=_EXPECTED)

    def test_only_the_latest_observation_is_judged(self) -> None:
        """A healthy earlier poll cannot mask a broken final one."""
        broken = [_cell_target(cell_index=0), _cell_target(cell_index=1, ready=False), _cell_target(kind="rollout")]

        with pytest.raises(AssertionError, match="not alive and ready"):
            assert_end_state_complete(
                [_observation(_complete(), at=_at(0)), _observation(broken, at=_at(1))],
                expected_count_of_kind=_EXPECTED,
            )
