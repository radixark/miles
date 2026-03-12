import math

from tests.fast.utils.ft.utils.metric_injectors import make_fake_mini_wandb

from miles.utils.ft.controller.detectors.checks.metrics import get_non_finite_loss


class TestGetNonFiniteLoss:
    def test_nan_returns_nan(self) -> None:
        wandb = make_fake_mini_wandb(steps={1: {"loss": float("nan")}})
        result = get_non_finite_loss(wandb)
        assert result is not None
        assert math.isnan(result)

    def test_inf_returns_inf(self) -> None:
        wandb = make_fake_mini_wandb(steps={1: {"loss": float("inf")}})
        result = get_non_finite_loss(wandb)
        assert result == float("inf")

    def test_finite_value_returns_none(self) -> None:
        wandb = make_fake_mini_wandb(steps={1: {"loss": 2.5}})
        assert get_non_finite_loss(wandb) is None

    def test_no_data_returns_none(self) -> None:
        wandb = make_fake_mini_wandb()
        assert get_non_finite_loss(wandb) is None

    def test_zero_is_finite(self) -> None:
        wandb = make_fake_mini_wandb(steps={1: {"loss": 0.0}})
        assert get_non_finite_loss(wandb) is None
