from datetime import datetime, timedelta, timezone

from tests.fast.utils.ft.helpers import (
    inject_gpu_temperature,
    make_detector_context,
    make_fake_metric_store,
    make_fake_mini_wandb,
)

from miles.utils.ft.controller.detectors.mfu_decline import MfuDeclineDetector
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType

_RANK_PLACEMENT = {0: "node-0", 1: "node-1"}


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
            run_id=run_id, rank=0, step=i + 1,
            metrics={"mfu": value}, receive_time=timestamp,
        )
    return wandb


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
        now = datetime.now(timezone.utc)
        low_mfu_entries: list[tuple[float, datetime]] = [
            (0.3, now - timedelta(minutes=40) + timedelta(minutes=i))
            for i in range(41)
        ]
        wandb = _make_wandb_with_timed_mfu(low_mfu_entries)

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

        decision = detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb, rank_placement=_RANK_PLACEMENT,
        ))

        assert decision.action == ActionType.NOTIFY_HUMAN

    def test_decline_timeout_not_yet_reached(self) -> None:
        now = datetime.now(timezone.utc)
        low_mfu_entries: list[tuple[float, datetime]] = [
            (0.3, now - timedelta(minutes=10) + timedelta(minutes=i))
            for i in range(11)
        ]
        wandb = _make_wandb_with_timed_mfu(low_mfu_entries)

        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=65.0)

        detector = MfuDeclineDetector(
            mfu_baseline=0.5,
            mfu_threshold_ratio=0.8,
            consecutive_steps=10,
            decline_timeout_minutes=30.0,
        )

        decision = detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb, rank_placement={0: "node-0"},
        ))

        assert decision.action == ActionType.NONE
        assert "monitoring" in decision.reason

    def test_dynamic_baseline(self) -> None:
        high_mfu = [0.5] * 50
        low_mfu = [0.3] * 10
        all_steps = high_mfu + low_mfu
        wandb = _make_wandb_with_mfu(all_steps)
        store = make_fake_metric_store()

        detector = MfuDeclineDetector(
            mfu_baseline=0.0,
            mfu_threshold_ratio=0.8,
            consecutive_steps=10,
        )

        decision = detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb,
        ))

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

        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=65.0)

        detector = MfuDeclineDetector(
            mfu_baseline=0.5,
            mfu_threshold_ratio=0.8,
            consecutive_steps=10,
            decline_timeout_minutes=30.0,
        )

        decision = detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb, rank_placement={0: "node-0"},
        ))

        assert decision.action == ActionType.NONE
        assert "monitoring" in decision.reason
