from tests.fast.utils.ft.helpers import make_detector_context, make_fake_mini_wandb

from miles.utils.ft.controller.detectors.nan_loss import NanLossDetector
from miles.utils.ft.models import ActionType


class TestNanLossDetector:
    def test_loss_normal(self) -> None:
        wandb = make_fake_mini_wandb(steps={1: {"loss": 2.5}})
        detector = NanLossDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE

    def test_loss_nan(self) -> None:
        wandb = make_fake_mini_wandb(steps={1: {"loss": float("nan")}})
        detector = NanLossDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "nan_loss"

    def test_loss_inf(self) -> None:
        wandb = make_fake_mini_wandb(steps={1: {"loss": float("inf")}})
        detector = NanLossDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "nan_loss"

    def test_loss_negative_inf(self) -> None:
        wandb = make_fake_mini_wandb(steps={1: {"loss": float("-inf")}})
        detector = NanLossDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "nan_loss"

    def test_no_loss_data(self) -> None:
        wandb = make_fake_mini_wandb()
        detector = NanLossDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE
