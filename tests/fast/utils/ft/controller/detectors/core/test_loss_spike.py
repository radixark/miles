import pytest

from tests.fast.utils.ft.utils import make_detector_context, make_fake_mini_wandb

from miles.utils.ft.controller.detectors.core.loss_spike import (
    LossSpikeDetector,
    LossSpikeDetectorConfig,
)
from miles.utils.ft.controller.types import ActionType, TriggerType


def _make_steps(baseline_steps: int, baseline_loss: float, recent_steps: int, recent_loss: float) -> dict:
    steps: dict[int, dict[str, float]] = {}
    for i in range(1, baseline_steps + 1):
        steps[i] = {"loss": baseline_loss}
    for i in range(baseline_steps + 1, baseline_steps + recent_steps + 1):
        steps[i] = {"loss": recent_loss}
    return steps


class TestLossSpikeDetector:
    def test_no_spike_normal_loss(self) -> None:
        steps = _make_steps(baseline_steps=50, baseline_loss=2.0, recent_steps=5, recent_loss=2.1)
        wandb = make_fake_mini_wandb(steps=steps)
        detector = LossSpikeDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE

    def test_5x_loss_spike_triggers_recovery(self) -> None:
        """A 5x increase in loss relative to baseline should trigger recovery.
        Previously only NaN/Inf was detected; gradual spikes (e.g. from SDC)
        that stayed finite were missed entirely."""
        steps = _make_steps(baseline_steps=50, baseline_loss=2.0, recent_steps=5, recent_loss=10.0)
        wandb = make_fake_mini_wandb(steps=steps)
        detector = LossSpikeDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == TriggerType.NAN_LOSS
        assert "loss" in decision.reason
        assert "5.0x" in decision.reason

    def test_spike_just_below_threshold_no_action(self) -> None:
        steps = _make_steps(baseline_steps=50, baseline_loss=2.0, recent_steps=5, recent_loss=9.9)
        wandb = make_fake_mini_wandb(steps=steps)
        detector = LossSpikeDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE

    def test_grad_norm_spike_triggers_recovery(self) -> None:
        """Grad norm spike detection — paper §4.1 mentions both loss and
        gradient norms as signals for abnormal metrics."""
        steps: dict[int, dict[str, float]] = {}
        for i in range(1, 51):
            steps[i] = {"loss": 2.0, "grad_norm": 1.0}
        for i in range(51, 56):
            steps[i] = {"loss": 2.0, "grad_norm": 6.0}
        wandb = make_fake_mini_wandb(steps=steps)
        detector = LossSpikeDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.ENTER_RECOVERY
        assert "grad_norm" in decision.reason

    def test_insufficient_data_no_action(self) -> None:
        wandb = make_fake_mini_wandb(steps={1: {"loss": 2.0}, 2: {"loss": 2.0}})
        detector = LossSpikeDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE

    def test_nan_in_window_skipped(self) -> None:
        """NaN values are handled by NanLossDetector; spike detector skips them."""
        steps = _make_steps(baseline_steps=50, baseline_loss=2.0, recent_steps=5, recent_loss=float("nan"))
        wandb = make_fake_mini_wandb(steps=steps)
        detector = LossSpikeDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE

    def test_custom_threshold(self) -> None:
        config = LossSpikeDetectorConfig(spike_threshold=3.0, recent_steps=5, baseline_steps=50)
        steps = _make_steps(baseline_steps=50, baseline_loss=2.0, recent_steps=5, recent_loss=6.0)
        wandb = make_fake_mini_wandb(steps=steps)
        detector = LossSpikeDetector(config=config)

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.ENTER_RECOVERY

    def test_zero_baseline_no_action(self) -> None:
        """Baseline avg <= 0 is ambiguous; skip detection to avoid division issues."""
        steps = _make_steps(baseline_steps=50, baseline_loss=0.0, recent_steps=5, recent_loss=5.0)
        wandb = make_fake_mini_wandb(steps=steps)
        detector = LossSpikeDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE

    def test_both_loss_and_grad_norm_spike_reported(self) -> None:
        steps: dict[int, dict[str, float]] = {}
        for i in range(1, 51):
            steps[i] = {"loss": 2.0, "grad_norm": 1.0}
        for i in range(51, 56):
            steps[i] = {"loss": 12.0, "grad_norm": 7.0}
        wandb = make_fake_mini_wandb(steps=steps)
        detector = LossSpikeDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.ENTER_RECOVERY
        assert "loss" in decision.reason
        assert "grad_norm" in decision.reason

    def test_missing_metric_does_not_fail(self) -> None:
        """If grad_norm is not logged, spike check should gracefully skip it."""
        steps: dict[int, dict[str, float]] = {}
        for i in range(1, 56):
            steps[i] = {"loss": 2.0}
        wandb = make_fake_mini_wandb(steps=steps)
        detector = LossSpikeDetector()

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE


class TestLossSpikeDetectorValidation:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(spike_threshold=1.0), "spike_threshold"),
            (dict(spike_threshold=0.5), "spike_threshold"),
            (dict(recent_steps=0), "recent_steps"),
            (dict(baseline_steps=0), "baseline_steps"),
        ],
    )
    def test_invalid_parameter_rejected(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            LossSpikeDetectorConfig(**kwargs)
