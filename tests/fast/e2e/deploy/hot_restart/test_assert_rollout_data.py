from pathlib import Path

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_rollout_data import assert_generations_recorded_their_steps
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import ScheduledFreeze

_SCHEDULE = (ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),)


class TestRolloutGenerationArtifacts:
    def test_generations_that_recorded_their_steps_pass(self, recorded_rollout_generations: list[str]) -> None:
        """Each generation directory holding exactly the steps it generated passes."""
        assert_generations_recorded_their_steps(recorded_rollout_generations, schedule=_SCHEDULE, num_rollouts=4)

    @pytest.mark.parametrize("name", ["generation_9", "stray.pt"])
    def test_unexpected_artifacts_still_fail(
        self, tmp_path: Path, recorded_rollout_generations: list[str], name: str
    ) -> None:
        """Output from a generation nothing relaunched fails the check."""
        if name.endswith(".pt"):
            (tmp_path / name).touch()
        else:
            (tmp_path / name).mkdir()

        with pytest.raises(AssertionError, match="beside"):
            assert_generations_recorded_their_steps(recorded_rollout_generations, schedule=_SCHEDULE, num_rollouts=4)

    def test_a_missing_generation_step_fails(self, tmp_path: Path, recorded_rollout_generations: list[str]) -> None:
        """A generation that did not record one of its steps fails the coverage check."""
        (tmp_path / "generation_1" / "3.pt").unlink()

        with pytest.raises(AssertionError, match="generation 1"):
            assert_generations_recorded_their_steps(recorded_rollout_generations, schedule=_SCHEDULE, num_rollouts=4)
