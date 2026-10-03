from pathlib import Path

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_rollout_data import assert_generations_recorded_their_steps
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import ScheduledFreeze

_SCHEDULE = (ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),)


class TestRolloutGenerationArtifacts:
    def test_dashboard_sidecar_does_not_count_as_an_unexpected_generation(
        self, recorded_rollout_generations: list[str]
    ) -> None:
        """The rollout writer emits dashboard columns beside valid generation directories."""
        assert_generations_recorded_their_steps(recorded_rollout_generations, schedule=_SCHEDULE, num_rollouts=4)

    @pytest.mark.parametrize("name", ["generation_9", "stray.pt"])
    def test_unexpected_artifacts_still_fail(
        self, tmp_path: Path, recorded_rollout_generations: list[str], name: str
    ) -> None:
        """Allowing dashboard columns must not hide output from an unexpected generation."""
        if name.endswith(".pt"):
            (tmp_path / name).touch()
        else:
            (tmp_path / name).mkdir()

        with pytest.raises(AssertionError, match="beside"):
            assert_generations_recorded_their_steps(recorded_rollout_generations, schedule=_SCHEDULE, num_rollouts=4)

    def test_dashboard_name_does_not_allow_a_stray_file(
        self, tmp_path: Path, recorded_rollout_generations: list[str]
    ) -> None:
        """Only the dashboard directory is a legitimate sidecar."""
        dashboard = tmp_path / "dashboard_columns"
        dashboard.rmdir()
        dashboard.touch()

        with pytest.raises(AssertionError, match="beside"):
            assert_generations_recorded_their_steps(recorded_rollout_generations, schedule=_SCHEDULE, num_rollouts=4)

    def test_missing_generation_step_still_fails_with_dashboard_columns(
        self, tmp_path: Path, recorded_rollout_generations: list[str]
    ) -> None:
        """The dashboard exception preserves exact rollout coverage checks."""
        (tmp_path / "generation_1" / "3.pt").unlink()

        with pytest.raises(AssertionError, match="generation 1"):
            assert_generations_recorded_their_steps(recorded_rollout_generations, schedule=_SCHEDULE, num_rollouts=4)
