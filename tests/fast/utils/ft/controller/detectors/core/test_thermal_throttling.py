import pytest

from tests.fast.utils.ft.helpers import (
    inject_gpu_temperature,
    make_detector_context,
    make_fake_metric_store,
    make_fake_mini_wandb,
)

from miles.utils.ft.controller.detectors.core.thermal_throttling import (
    ThermalThrottlingDetector,
    ThermalThrottlingDetectorConfig,
)
from miles.utils.ft.models.fault import ActionType, TriggerType

_RANK_PLACEMENT = {0: "node-0", 1: "node-1"}


class TestThermalThrottlingDetector:
    def test_no_temperature_data(self) -> None:
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.45} for i in range(1, 11)})
        detector = ThermalThrottlingDetector()

        decision = detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb, rank_placement=_RANK_PLACEMENT,
        ))

        assert decision.action == ActionType.NONE

    def test_uniform_temperatures_no_outlier(self) -> None:
        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=65.0)
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=66.0)

        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.3} for i in range(1, 11)})
        detector = ThermalThrottlingDetector()

        decision = detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb, rank_placement=_RANK_PLACEMENT,
        ))

        assert decision.action == ActionType.NONE

    def test_temperature_outlier_with_mfu_decline_triggers_recovery(self) -> None:
        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=60.0)
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=105.0)

        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.3} for i in range(1, 61)})
        detector = ThermalThrottlingDetector(config=ThermalThrottlingDetectorConfig(
            temperature_delta_threshold=20.0,
            mfu_decline_threshold_ratio=0.9,
            mfu_baseline=0.5,
            mfu_consecutive_steps=10,
        ))

        decision = detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb, rank_placement=_RANK_PLACEMENT,
        ))

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == TriggerType.HARDWARE
        assert "node-1" in decision.bad_node_ids
        assert "thermal throttling" in decision.reason

    def test_temperature_outlier_but_mfu_healthy_no_action(self) -> None:
        """Temperature outlier alone does not trigger recovery without MFU decline."""
        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=60.0)
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=105.0)

        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.48} for i in range(1, 61)})
        detector = ThermalThrottlingDetector(config=ThermalThrottlingDetectorConfig(
            temperature_delta_threshold=20.0,
            mfu_decline_threshold_ratio=0.9,
            mfu_baseline=0.5,
            mfu_consecutive_steps=10,
        ))

        decision = detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb, rank_placement=_RANK_PLACEMENT,
        ))

        assert decision.action == ActionType.NONE
        assert "MFU is healthy" in decision.reason

    def test_empty_rank_placement(self) -> None:
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb()
        detector = ThermalThrottlingDetector()

        decision = detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb, rank_placement={},
        ))

        assert decision.action == ActionType.NONE

    def test_picks_hottest_node_when_multiple_outliers(self) -> None:
        """When multiple nodes exceed the delta threshold, the hottest is reported."""
        store = make_fake_metric_store()
        rank_placement = {0: "node-0", 1: "node-1", 2: "node-2"}
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=60.0)
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=95.0)
            inject_gpu_temperature(store, node_id="node-2", gpu=str(i), celsius=110.0)

        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.3} for i in range(1, 61)})
        detector = ThermalThrottlingDetector(config=ThermalThrottlingDetectorConfig(
            temperature_delta_threshold=5.0,
            mfu_baseline=0.5,
            mfu_consecutive_steps=10,
        ))

        decision = detector.evaluate(make_detector_context(
            metric_store=store, mini_wandb=wandb, rank_placement=rank_placement,
        ))

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.bad_node_ids == ["node-2"]


class TestThermalThrottlingDetectorValidation:
    @pytest.mark.parametrize("kwargs,match", [
        (dict(temperature_delta_threshold=0.0), "temperature_delta_threshold"),
        (dict(temperature_delta_threshold=-1.0), "temperature_delta_threshold"),
        (dict(mfu_decline_threshold_ratio=0.0), "mfu_decline_threshold_ratio"),
        (dict(mfu_decline_threshold_ratio=1.5), "mfu_decline_threshold_ratio"),
        (dict(mfu_consecutive_steps=0), "mfu_consecutive_steps"),
        (dict(mfu_baseline_steps=0), "mfu_baseline_steps"),
    ])
    def test_invalid_parameter_rejected(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ThermalThrottlingDetectorConfig(**kwargs)
