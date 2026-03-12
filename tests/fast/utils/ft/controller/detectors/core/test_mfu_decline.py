from datetime import datetime, timedelta, timezone

import pytest
from tests.fast.utils.ft.utils import make_detector_context, make_fake_mini_wandb

from miles.utils.ft.controller.detectors.core.mfu_decline import MfuDeclineDetector, MfuDeclineDetectorConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.types import ActionType


def _make_wandb_with_mfu(
    mfu_values: list[float],
    start_step: int = 1,
) -> MiniWandb:
    steps = {start_step + i: {"mfu": v} for i, v in enumerate(mfu_values)}
    return make_fake_mini_wandb(steps=steps)


def _make_wandb_with_timed_mfu(
    entries: list[tuple[float, datetime]],
    run_id: str = "test-run",
) -> MiniWandb:
    """Create a MiniWandb where each MFU value has an explicit receive_time."""
    wandb = MiniWandb(active_run_id=run_id)
    for i, (value, timestamp) in enumerate(entries):
        wandb.log_step(
            run_id=run_id,
            step=i + 1,
            metrics={"mfu": value},
            receive_time=timestamp,
        )
    return wandb


class TestMfuDeclineDetector:
    def test_normal_mfu(self) -> None:
        wandb = _make_wandb_with_mfu([0.45] * 10)
        detector = MfuDeclineDetector(
            config=MfuDeclineDetectorConfig(
                mfu_baseline=0.5,
                mfu_threshold_ratio=0.8,
                consecutive_steps=10,
            )
        )

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE

    def test_insufficient_data(self) -> None:
        wandb = _make_wandb_with_mfu([0.1] * 5)
        detector = MfuDeclineDetector(
            config=MfuDeclineDetectorConfig(
                mfu_baseline=0.5,
                consecutive_steps=10,
            )
        )

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE

    def test_decline_monitoring(self) -> None:
        wandb = _make_wandb_with_mfu([0.3] * 10)
        detector = MfuDeclineDetector(
            config=MfuDeclineDetectorConfig(
                mfu_baseline=0.5,
                mfu_threshold_ratio=0.8,
                consecutive_steps=10,
            )
        )

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE
        assert "monitoring" in decision.reason

    def test_decline_timeout_notify_human(self) -> None:
        now = datetime.now(timezone.utc)
        low_mfu_entries: list[tuple[float, datetime]] = [
            (0.3, now - timedelta(minutes=40) + timedelta(minutes=i)) for i in range(41)
        ]
        wandb = _make_wandb_with_timed_mfu(low_mfu_entries)

        detector = MfuDeclineDetector(
            config=MfuDeclineDetectorConfig(
                mfu_baseline=0.5,
                mfu_threshold_ratio=0.8,
                consecutive_steps=10,
                decline_timeout_minutes=30.0,
            )
        )

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NOTIFY_HUMAN

    def test_decline_timeout_not_yet_reached(self) -> None:
        now = datetime.now(timezone.utc)
        low_mfu_entries: list[tuple[float, datetime]] = [
            (0.3, now - timedelta(minutes=10) + timedelta(minutes=i)) for i in range(11)
        ]
        wandb = _make_wandb_with_timed_mfu(low_mfu_entries)

        detector = MfuDeclineDetector(
            config=MfuDeclineDetectorConfig(
                mfu_baseline=0.5,
                mfu_threshold_ratio=0.8,
                consecutive_steps=10,
                decline_timeout_minutes=30.0,
            )
        )

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE
        assert "monitoring" in decision.reason

    def test_dynamic_baseline(self) -> None:
        wandb = _make_wandb_with_mfu([0.5] * 50 + [0.3] * 10)

        detector = MfuDeclineDetector(
            config=MfuDeclineDetectorConfig(
                mfu_baseline=None,
                mfu_threshold_ratio=0.8,
                consecutive_steps=10,
            )
        )

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE
        assert "monitoring" in decision.reason

    def test_mfu_recovery_resets_decline_window(self) -> None:
        """If MFU recovers mid-window, the decline timer restarts from the
        last healthy reading, so a brief dip doesn't trigger NOTIFY_HUMAN."""
        now = datetime.now(timezone.utc)
        entries: list[tuple[float, datetime]] = [
            *[(0.3, now - timedelta(minutes=35) + timedelta(minutes=i)) for i in range(20)],
            (0.48, now - timedelta(minutes=15)),
            *[(0.3, now - timedelta(minutes=14) + timedelta(minutes=i)) for i in range(15)],
        ]
        wandb = _make_wandb_with_timed_mfu(entries)

        detector = MfuDeclineDetector(
            config=MfuDeclineDetectorConfig(
                mfu_baseline=0.5,
                mfu_threshold_ratio=0.8,
                consecutive_steps=10,
                decline_timeout_minutes=30.0,
            )
        )

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NONE
        assert "monitoring" in decision.reason


class TestMfuAbsoluteMinimum:
    def test_triggers_when_below_floor(self) -> None:
        wandb = _make_wandb_with_mfu([0.05] * 10)
        detector = MfuDeclineDetector(
            config=MfuDeclineDetectorConfig(
                mfu_baseline=0.5,
                mfu_absolute_minimum=0.1,
                consecutive_steps=10,
            )
        )

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert decision.action == ActionType.NOTIFY_HUMAN
        assert "absolute minimum" in decision.reason

    def test_disabled_by_default(self) -> None:
        wandb = _make_wandb_with_mfu([0.05] * 10)
        detector = MfuDeclineDetector(
            config=MfuDeclineDetectorConfig(
                mfu_baseline=0.5,
                consecutive_steps=10,
            )
        )

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert "absolute minimum" not in decision.reason

    def test_no_trigger_when_above_floor(self) -> None:
        wandb = _make_wandb_with_mfu([0.45] * 10)
        detector = MfuDeclineDetector(
            config=MfuDeclineDetectorConfig(
                mfu_baseline=0.5,
                mfu_absolute_minimum=0.1,
                consecutive_steps=10,
            )
        )

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        assert "absolute minimum" not in decision.reason


class TestMfuDeclineNoiseSpikeResistance:
    def test_single_noise_spike_does_not_reset_decline_timer(self) -> None:
        """_compute_decline_duration_minutes previously used per-step raw MFU
        values to find the last healthy reading, so a single noise spike above
        threshold would reset the decline timer. Now it uses a sliding window
        average (same as check_mfu_health), making it resistant to noise."""
        now = datetime.now(timezone.utc)
        entries: list[tuple[float, datetime]] = []

        # 100 steps, all at 0.19 (below threshold 0.20), except step 50 = 0.21 (noise spike)
        for i in range(100):
            mfu_val = 0.21 if i == 50 else 0.19
            entries.append((mfu_val, now - timedelta(minutes=100 - i)))
        wandb = _make_wandb_with_timed_mfu(entries)

        detector = MfuDeclineDetector(
            config=MfuDeclineDetectorConfig(
                mfu_baseline=0.25,
                mfu_threshold_ratio=0.8,  # threshold = 0.20
                consecutive_steps=10,
                decline_timeout_minutes=60.0,
            )
        )

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        # With sliding average, a single spike in 100 below-threshold steps
        # should not prevent the timeout from triggering
        assert decision.action == ActionType.NOTIFY_HUMAN

    def test_genuine_recovery_resets_decline_timer_with_sliding_average(self) -> None:
        """A sustained recovery period (not just a spike) still resets the timer."""
        now = datetime.now(timezone.utc)
        entries: list[tuple[float, datetime]] = []

        # 20 steps below threshold, then 15 steps above threshold (genuine recovery), then 20 below again
        for i in range(20):
            entries.append((0.19, now - timedelta(minutes=55 - i)))
        for i in range(15):
            entries.append((0.25, now - timedelta(minutes=35 - i)))
        for i in range(20):
            entries.append((0.19, now - timedelta(minutes=20 - i)))
        wandb = _make_wandb_with_timed_mfu(entries)

        detector = MfuDeclineDetector(
            config=MfuDeclineDetectorConfig(
                mfu_baseline=0.25,
                mfu_threshold_ratio=0.8,  # threshold = 0.20
                consecutive_steps=10,
                decline_timeout_minutes=30.0,
            )
        )

        decision = detector.evaluate(make_detector_context(mini_wandb=wandb))

        # Genuine recovery resets timer; only 20min of decline since recovery → not yet timed out
        assert decision.action == ActionType.NONE
        assert "monitoring" in decision.reason


class TestMfuDeclineDetectorValidation:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(mfu_threshold_ratio=0.0), "mfu_threshold_ratio"),
            (dict(mfu_threshold_ratio=-0.5), "mfu_threshold_ratio"),
            (dict(mfu_threshold_ratio=1.5), "mfu_threshold_ratio"),
            (dict(consecutive_steps=0), "consecutive_steps"),
            (dict(decline_timeout_minutes=0.0), "decline_timeout_minutes"),
            (dict(baseline_steps=0), "baseline_steps"),
            (dict(mfu_absolute_minimum=-0.1), "mfu_absolute_minimum"),
        ],
    )
    def test_invalid_parameter_rejected(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            MfuDeclineDetectorConfig(**kwargs)
