import math

from tests.fast.utils.ft.utils import make_fake_mini_wandb

from miles.utils.ft.controller.detectors.checks.mfu_health import check_mfu_health


class TestCheckMfuHealth:
    def test_insufficient_data_returns_none(self) -> None:
        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.5} for i in range(1, 6)})

        result = check_mfu_health(
            wandb,
            consecutive_steps=10,
            threshold_ratio=0.8,
            baseline=0.5,
            baseline_steps=50,
        )

        assert result is None

    def test_explicit_baseline(self) -> None:
        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.45} for i in range(1, 11)})

        result = check_mfu_health(
            wandb,
            consecutive_steps=10,
            threshold_ratio=0.8,
            baseline=0.5,
            baseline_steps=50,
        )

        assert result is not None
        assert result.baseline == 0.5
        assert result.threshold == 0.4
        assert abs(result.avg_mfu - 0.45) < 1e-9
        assert not result.is_declining

    def test_declining_mfu(self) -> None:
        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.3} for i in range(1, 11)})

        result = check_mfu_health(
            wandb,
            consecutive_steps=10,
            threshold_ratio=0.8,
            baseline=0.5,
            baseline_steps=50,
        )

        assert result is not None
        assert result.is_declining

    def test_dynamic_baseline_from_history(self) -> None:
        steps = {i: {"mfu": 0.5} for i in range(1, 51)}
        steps.update({i: {"mfu": 0.3} for i in range(51, 61)})
        wandb = make_fake_mini_wandb(steps=steps)

        result = check_mfu_health(
            wandb,
            consecutive_steps=10,
            threshold_ratio=0.8,
            baseline=None,
            baseline_steps=50,
        )

        assert result is not None
        assert abs(result.baseline - 0.5) < 1e-9
        assert result.is_declining

    def test_no_baseline_data_returns_none(self) -> None:
        """When only recent steps exist and no explicit baseline, returns None."""
        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.3} for i in range(1, 11)})

        result = check_mfu_health(
            wandb,
            consecutive_steps=10,
            threshold_ratio=0.8,
            baseline=None,
            baseline_steps=50,
        )

        assert result is None


class TestNonFiniteMfuValues:
    """NaN/Inf MFU values used to propagate through arithmetic unchecked,
    making avg_mfu=NaN. Since NaN < threshold is always False, both
    MfuDeclineDetector and ThermalThrottlingDetector would incorrectly
    report "MFU within acceptable range" / "MFU is healthy"."""

    def test_nan_in_recent_window_marks_telemetry_invalid(self) -> None:
        steps: dict[int, dict[str, float]] = {i: {"mfu": 0.5} for i in range(1, 10)}
        steps[10] = {"mfu": float("nan")}
        wandb = make_fake_mini_wandb(steps=steps)

        result = check_mfu_health(
            wandb,
            consecutive_steps=10,
            threshold_ratio=0.8,
            baseline=0.5,
            baseline_steps=50,
        )

        assert result is not None
        assert not result.telemetry_valid
        assert math.isnan(result.avg_mfu)

    def test_inf_in_recent_window_marks_telemetry_invalid(self) -> None:
        steps: dict[int, dict[str, float]] = {i: {"mfu": 0.5} for i in range(1, 10)}
        steps[10] = {"mfu": float("inf")}
        wandb = make_fake_mini_wandb(steps=steps)

        result = check_mfu_health(
            wandb,
            consecutive_steps=10,
            threshold_ratio=0.8,
            baseline=0.5,
            baseline_steps=50,
        )

        assert result is not None
        assert not result.telemetry_valid

    def test_nan_in_baseline_window_marks_telemetry_invalid(self) -> None:
        steps: dict[int, dict[str, float]] = {i: {"mfu": 0.5} for i in range(1, 51)}
        steps[25] = {"mfu": float("nan")}
        steps.update({i: {"mfu": 0.3} for i in range(51, 61)})
        wandb = make_fake_mini_wandb(steps=steps)

        result = check_mfu_health(
            wandb,
            consecutive_steps=10,
            threshold_ratio=0.8,
            baseline=None,
            baseline_steps=50,
        )

        assert result is not None
        assert not result.telemetry_valid

    def test_all_finite_values_remain_valid(self) -> None:
        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.5} for i in range(1, 11)})

        result = check_mfu_health(
            wandb,
            consecutive_steps=10,
            threshold_ratio=0.8,
            baseline=0.5,
            baseline_steps=50,
        )

        assert result is not None
        assert result.telemetry_valid
