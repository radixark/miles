import pytest
from pydantic import ValidationError
from tests.utils.soak.core.config import SoakRunnerConfig, SoakTailConfig, SoakTargetConfig, SoakTimeoutConfig


class TestSoakTargetConfig:
    @pytest.mark.parametrize(
        "fields",
        [
            {"expected_count": 0, "mean_interval_seconds": 1.0},
            {"expected_count": 1, "mean_interval_seconds": 0.0},
            {"expected_count": 1, "mean_interval_seconds": float("inf")},
            {"expected_count": 1, "mean_interval_seconds": float("nan")},
        ],
    )
    def test_a_target_without_replicas_or_a_finite_positive_interval_is_rejected(self, fields: dict) -> None:
        """A kind needs at least one target and a finite positive mean injection interval."""
        with pytest.raises(ValidationError):
            SoakTargetConfig(**fields)


class TestSoakRunnerConfig:
    def test_an_unknown_field_is_rejected(self) -> None:
        """A misspelled runner option must fail instead of silently falling back to a default."""
        with pytest.raises(ValidationError):
            SoakRunnerConfig(seed=1, quiescent_poll_required=3)

    def test_the_config_round_trips_through_json_with_nested_policies(self) -> None:
        """The run context persists the runner config, so JSON must restore it exactly."""
        config = SoakRunnerConfig(
            seed=7,
            start_after_rollout_id=2,
            target_configs={"actor": SoakTargetConfig(expected_count=2, mean_interval_seconds=30.0)},
            timeouts=SoakTimeoutConfig(run_seconds=10.0),
            tail=SoakTailConfig(close_after_rollout_id=5),
            poll_interval_seconds=0.5,
            quiescent_polls_required=3,
        )

        assert SoakRunnerConfig.model_validate_json(config.model_dump_json()) == config

    @pytest.mark.parametrize(
        "fields",
        [
            {"poll_interval_seconds": 0.0},
            {"quiescent_polls_required": 0},
            {"start_after_rollout_id": -1},
            {"timeouts": {"tail_seconds": 0.0}},
            {"timeouts": {"run_seconds": float("inf")}},
        ],
    )
    def test_nonpositive_or_infinite_limits_are_rejected(self, fields: dict) -> None:
        """Zero polls, zero intervals, negative starts and unbounded timeouts are invalid soak limits."""
        with pytest.raises(ValidationError):
            SoakRunnerConfig(seed=0, **fields)


class TestSoakTailConfigCreate:
    @pytest.mark.parametrize(
        ("num_rollout", "close_after_rollout_id"),
        [(4, 0), (10, 6), (15, 11), (20, 15), (100, 79)],
    )
    def test_admission_closes_leaving_the_larger_of_three_rollouts_and_a_fifth_of_the_run(
        self, num_rollout: int, close_after_rollout_id: int
    ) -> None:
        """The tail keeps max(3, num_rollout // 5) rollouts after the last admitted one."""
        assert SoakTailConfig.create(num_rollout=num_rollout).close_after_rollout_id == close_after_rollout_id

    @pytest.mark.parametrize(("num_rollout", "min_tail_rollouts"), [(3, 3), (1, 3), (10, 2), (5, 5)])
    def test_a_run_too_short_for_a_recovery_tail_is_rejected(self, num_rollout: int, min_tail_rollouts: int) -> None:
        """A tail shorter than three rollouts or covering the whole run cannot prove recovery."""
        with pytest.raises(ValueError, match="complete recovery tail"):
            SoakTailConfig.create(num_rollout=num_rollout, min_tail_rollouts=min_tail_rollouts)

    def test_a_larger_minimum_tail_moves_the_closing_rollout_earlier(self) -> None:
        """An explicit minimum tail wins over the fifth-of-run default when it is longer."""
        assert SoakTailConfig.create(num_rollout=20, min_tail_rollouts=6).close_after_rollout_id == 13
