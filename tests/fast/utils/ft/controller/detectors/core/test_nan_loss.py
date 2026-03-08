import pytest

from tests.fast.utils.ft.utils import make_detector_context, make_fake_mini_wandb

from miles.utils.ft.controller.detectors.core.nan_loss import NanLossDetector
from miles.utils.ft.models.fault import ActionType


class TestNanLossDetector:
    def test_loss_normal(self) -> None:
        wandb = make_fake_mini_wandb(steps={1: {"loss": 2.5}})
        detector = NanLossDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE

    @pytest.mark.parametrize("bad_loss", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "neg_inf"])
    def test_non_finite_loss_triggers_recovery(self, bad_loss: float) -> None:
        wandb = make_fake_mini_wandb(steps={1: {"loss": bad_loss}})
        detector = NanLossDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "nan_loss"

    def test_no_loss_data(self) -> None:
        wandb = make_fake_mini_wandb()
        detector = NanLossDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE
