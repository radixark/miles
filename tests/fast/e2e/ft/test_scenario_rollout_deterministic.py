from datetime import datetime, timedelta, timezone

from tests.e2e.ft.conftest_ft.scenario_rollout_deterministic import _compute_crashed_rollouts

_BASE = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


class TestComputeCrashedRollouts:
    def test_two_crashes_before_any_rollout_finished_share_one_window(self) -> None:
        """Both landing before the first generation is the vacuous run this scenario has to reject."""
        crashed = _compute_crashed_rollouts(
            injected_at=[_BASE, _BASE + timedelta(seconds=10)], rollout_completions=[(0, _BASE + timedelta(hours=1))]
        )

        assert crashed == {0}

    def test_crashes_on_either_side_of_a_finished_rollout_are_two_windows(self) -> None:
        """This is the run the scenario is meant to produce: crashes spread across the loss curve."""
        crashed = _compute_crashed_rollouts(
            injected_at=[_BASE, _BASE + timedelta(seconds=20)],
            rollout_completions=[(0, _BASE + timedelta(seconds=10))],
        )

        assert crashed == {0, 1}

    def test_a_run_with_no_crashes_has_no_windows(self) -> None:
        """An empty result must not read as coverage; the caller's floor is what rejects it."""
        assert _compute_crashed_rollouts(injected_at=[], rollout_completions=[(0, _BASE)]) == set()
