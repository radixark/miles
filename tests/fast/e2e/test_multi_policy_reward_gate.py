import pytest
from tests.e2e.conftest_multi_policy import _compute_reward_window_means


class TestComputeRewardWindowMeans:
    def test_the_early_value_averages_the_first_three_steps(self):
        """One noisy first rollout cannot decide whether training began unsolved."""
        windows = _compute_reward_window_means([0.9, 0.0, 0.3, 0.6, 0.9, 0.9])

        assert windows.initial == pytest.approx(0.4)

    def test_the_final_value_averages_the_last_third(self):
        """The learning gate measures a sustained tail rather than one lucky final rollout."""
        windows = _compute_reward_window_means([0.0, 0.1, 0.2, 0.3, 0.8, 1.0])

        assert windows.final == pytest.approx(0.9)

    def test_fewer_than_three_points_cannot_define_the_early_window(self):
        """A shortened run must not silently weaken the three-step early baseline."""
        with pytest.raises(AssertionError, match="at least three raw reward points"):
            _compute_reward_window_means([0.1, 0.2])
