from datetime import datetime, timedelta, timezone

from tests.fast.utils.ft.conftest import (
    inject_gpu_temperature,
    make_fake_metric_store,
    make_fake_mini_wandb,
)

from miles.utils.ft.controller.detectors.mfu_decline import MfuDeclineDetector
from miles.utils.ft.models import ActionType

_RANK_PLACEMENT = {0: "node-0", 1: "node-1"}


def _make_wandb_with_mfu(
    mfu_values: list[float],
    start_step: int = 1,
) -> object:
    steps = {start_step + i: {"mfu": v} for i, v in enumerate(mfu_values)}
    return make_fake_mini_wandb(steps=steps)


class TestMfuDeclineDetector:
    def test_normal_mfu(self) -> None:
        wandb = _make_wandb_with_mfu([0.45] * 10)
        detector = MfuDeclineDetector(
            mfu_baseline=0.5,
            mfu_threshold_ratio=0.8,
            consecutive_steps=10,
        )

        decision = detector.evaluate(make_fake_metric_store(), wandb, _RANK_PLACEMENT)

        assert decision.action == ActionType.NONE

    def test_insufficient_data(self) -> None:
        wandb = _make_wandb_with_mfu([0.1] * 5)
        detector = MfuDeclineDetector(
            mfu_baseline=0.5,
            consecutive_steps=10,
        )

        decision = detector.evaluate(make_fake_metric_store(), wandb, _RANK_PLACEMENT)

        assert decision.action == ActionType.NONE

    def test_decline_with_high_temperature(self) -> None:
        wandb = _make_wandb_with_mfu([0.3] * 10)
        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=60.0)
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=105.0)

        detector = MfuDeclineDetector(
            mfu_baseline=0.5,
            mfu_threshold_ratio=0.8,
            consecutive_steps=10,
            temperature_delta_threshold=20.0,
        )

        decision = detector.evaluate(store, wandb, _RANK_PLACEMENT)

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-1" in decision.bad_node_ids

    def test_decline_normal_temperature_monitoring(self) -> None:
        wandb = _make_wandb_with_mfu([0.3] * 10)
        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=65.0)
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=66.0)

        detector = MfuDeclineDetector(
            mfu_baseline=0.5,
            mfu_threshold_ratio=0.8,
            consecutive_steps=10,
        )

        decision = detector.evaluate(store, wandb, _RANK_PLACEMENT)

        assert decision.action == ActionType.NONE
        assert "monitoring" in decision.reason

    def test_decline_timeout_notify_human(self) -> None:
        wandb = _make_wandb_with_mfu([0.3] * 10)
        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=65.0)
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=66.0)

        detector = MfuDeclineDetector(
            mfu_baseline=0.5,
            mfu_threshold_ratio=0.8,
            consecutive_steps=10,
            decline_timeout_minutes=30.0,
        )

        # First call starts the timer
        detector.evaluate(store, wandb, _RANK_PLACEMENT)

        # Simulate time passing beyond timeout
        past_start = datetime.now(timezone.utc) - timedelta(minutes=35)
        detector._decline_start_time = past_start

        decision = detector.evaluate(store, wandb, _RANK_PLACEMENT)

        assert decision.action == ActionType.NOTIFY_HUMAN

    def test_dynamic_baseline(self) -> None:
        # 50 steps of high mfu followed by 10 steps of low mfu
        # Dynamic baseline = avg of last 50 steps = (40*0.5 + 10*0.3)/50 = 0.46
        # Threshold = 0.46 * 0.8 = 0.368, avg of last 10 = 0.3 < 0.368 → decline
        high_mfu = [0.5] * 50
        low_mfu = [0.3] * 10
        all_steps = high_mfu + low_mfu
        wandb = _make_wandb_with_mfu(all_steps)
        store = make_fake_metric_store()

        detector = MfuDeclineDetector(
            mfu_baseline=0.0,  # dynamic baseline
            mfu_threshold_ratio=0.8,
            consecutive_steps=10,
        )

        decision = detector.evaluate(store, wandb, {})

        assert decision.action == ActionType.NONE
        assert "monitoring" in decision.reason

    def test_mfu_recovers_resets_timer(self) -> None:
        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=65.0)

        detector = MfuDeclineDetector(
            mfu_baseline=0.5,
            mfu_threshold_ratio=0.8,
            consecutive_steps=10,
        )

        # First: MFU declining
        wandb_low = _make_wandb_with_mfu([0.3] * 10)
        detector.evaluate(store, wandb_low, {0: "node-0"})
        assert detector._decline_start_time is not None

        # Then: MFU recovers
        wandb_high = _make_wandb_with_mfu([0.45] * 10)
        detector.evaluate(store, wandb_high, {0: "node-0"})
        assert detector._decline_start_time is None
