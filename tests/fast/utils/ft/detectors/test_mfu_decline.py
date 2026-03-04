from datetime import datetime, timedelta, timezone

from tests.fast.utils.ft.conftest import (
    inject_gpu_temperature,
    make_detector_context,
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

        decision = detector.evaluate(make_detector_context(
            mini_wandb=wandb, rank_placement=_RANK_PLACEMENT,
        ))

        assert decision.action == ActionType.NONE

    def test_insufficient_data(self) -> None:
        wandb = _make_wandb_with_mfu([0.1] * 5)
        detector = MfuDeclineDetector(
            mfu_baseline=0.5,
            consecutive_steps=10,
        )

        decision = detector.evaluate(make_detector_context(
            mini_wandb=wandb, rank_placement=_RANK_PLACEMENT,
        ))

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

        decision = detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb, rank_placement=_RANK_PLACEMENT,
        ))

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

        decision = detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb, rank_placement=_RANK_PLACEMENT,
        ))

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

        ctx = make_detector_context(
            metric_store=store, mini_wandb=wandb, rank_placement=_RANK_PLACEMENT,
        )
        detector.evaluate(ctx)

        past_start = datetime.now(timezone.utc) - timedelta(minutes=35)
        detector._decline_start_time = past_start

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NOTIFY_HUMAN

    def test_dynamic_baseline(self) -> None:
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

        decision = detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb,
        ))

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

        wandb_low = _make_wandb_with_mfu([0.3] * 10)
        detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb_low, rank_placement={0: "node-0"},
        ))
        assert detector._decline_start_time is not None

        wandb_high = _make_wandb_with_mfu([0.45] * 10)
        detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb_high, rank_placement={0: "node-0"},
        ))
        assert detector._decline_start_time is None
