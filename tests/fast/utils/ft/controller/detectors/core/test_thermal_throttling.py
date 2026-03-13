import pytest

from tests.fast.utils.ft.utils import (
    inject_gpu_temperature,
    make_detector_context,
    make_fake_metric_store,
    make_fake_mini_wandb,
)

from miles.utils.ft.controller.detectors.core.thermal_throttling import (
    ThermalThrottlingDetector,
    ThermalThrottlingDetectorConfig,
)
from miles.utils.ft.controller.types import ActionType, TriggerType

_ACTIVE_NODE_IDS = {"node-0", "node-1"}


class TestThermalThrottlingDetector:
    def test_no_temperature_data(self) -> None:
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.45} for i in range(1, 11)})
        detector = ThermalThrottlingDetector()

        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                mini_wandb=wandb,
                active_node_ids=_ACTIVE_NODE_IDS,
            )
        )

        assert decision.action == ActionType.NONE

    def test_uniform_temperatures_no_outlier(self) -> None:
        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=65.0)
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=66.0)

        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.3} for i in range(1, 11)})
        detector = ThermalThrottlingDetector()

        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                mini_wandb=wandb,
                active_node_ids=_ACTIVE_NODE_IDS,
            )
        )

        assert decision.action == ActionType.NONE

    def test_temperature_outlier_with_mfu_decline_triggers_recovery(self) -> None:
        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=60.0)
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=105.0)

        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.3} for i in range(1, 61)})
        detector = ThermalThrottlingDetector(
            config=ThermalThrottlingDetectorConfig(
                temperature_delta_threshold=20.0,
                mfu_decline_threshold_ratio=0.9,
                mfu_baseline=0.5,
                mfu_consecutive_steps=10,
            )
        )

        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                mini_wandb=wandb,
                active_node_ids=_ACTIVE_NODE_IDS,
            )
        )

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == TriggerType.HARDWARE
        assert "node-1" in decision.bad_node_ids
        assert "thermal throttling" in decision.reason
        # Per-GPU detail should be included in the reason
        assert "gpu" in decision.reason

    def test_temperature_outlier_but_mfu_healthy_no_action(self) -> None:
        """Temperature outlier alone does not trigger recovery without MFU decline."""
        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=60.0)
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=105.0)

        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.48} for i in range(1, 61)})
        detector = ThermalThrottlingDetector(
            config=ThermalThrottlingDetectorConfig(
                temperature_delta_threshold=20.0,
                mfu_decline_threshold_ratio=0.9,
                mfu_baseline=0.5,
                mfu_consecutive_steps=10,
            )
        )

        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                mini_wandb=wandb,
                active_node_ids=_ACTIVE_NODE_IDS,
            )
        )

        assert decision.action == ActionType.NONE
        assert "MFU is healthy" in decision.reason

    def test_empty_active_node_ids(self) -> None:
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb()
        detector = ThermalThrottlingDetector()

        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                mini_wandb=wandb,
                active_node_ids=set(),
            )
        )

        assert decision.action == ActionType.NONE

    def test_all_outlier_nodes_returned_when_multiple_exceed_threshold(self) -> None:
        """All nodes exceeding the delta threshold are reported, not just the hottest."""
        store = make_fake_metric_store()
        active_node_ids = {"node-0", "node-1", "node-2"}
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=60.0)
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=95.0)
            inject_gpu_temperature(store, node_id="node-2", gpu=str(i), celsius=110.0)

        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.3} for i in range(1, 61)})
        detector = ThermalThrottlingDetector(
            config=ThermalThrottlingDetectorConfig(
                temperature_delta_threshold=5.0,
                mfu_baseline=0.5,
                mfu_consecutive_steps=10,
            )
        )

        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                mini_wandb=wandb,
                active_node_ids=active_node_ids,
            )
        )

        assert decision.action == ActionType.ENTER_RECOVERY
        assert set(decision.bad_node_ids) == {"node-1", "node-2"}
        assert decision.bad_node_ids[0] == "node-2", "hottest node should be listed first"

    def test_single_gpu_overheat_detected(self) -> None:
        """A single GPU overheating on a node is detected even when the other
        GPUs on that node are at normal temperature. Per-GPU detection
        compares each GPU individually to the cluster average, so a single
        100°C GPU among seven 65°C GPUs is caught (previously, a node-level
        mean would mask the outlier)."""
        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=65.0)
            temp = 100.0 if i == 0 else 65.0
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=temp)

        wandb = make_fake_mini_wandb(steps={i: {"mfu": 0.3} for i in range(1, 61)})
        detector = ThermalThrottlingDetector(
            config=ThermalThrottlingDetectorConfig(
                temperature_delta_threshold=20.0,
                mfu_decline_threshold_ratio=0.9,
                mfu_baseline=0.5,
                mfu_consecutive_steps=10,
            )
        )

        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                mini_wandb=wandb,
                active_node_ids=_ACTIVE_NODE_IDS,
            )
        )

        assert decision.action == ActionType.ENTER_RECOVERY
        assert "node-1" in decision.bad_node_ids
        # Per-GPU detail identifies the specific hot GPU
        assert "gpu0" in decision.reason
        assert "100" in decision.reason


class TestThermalThrottlingWithNonFiniteMfu:
    """When MFU data contained NaN, the thermal throttling detector would
    get mfu.is_declining=False (NaN < threshold is always False), and
    incorrectly report "temperature outlier but MFU is healthy"."""

    def test_temperature_outlier_with_nan_mfu_returns_telemetry_blind(self) -> None:
        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=60.0)
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=105.0)

        wandb = make_fake_mini_wandb(steps={i: {"mfu": float("nan")} for i in range(1, 11)})
        detector = ThermalThrottlingDetector(
            config=ThermalThrottlingDetectorConfig(
                temperature_delta_threshold=20.0,
                mfu_baseline=0.5,
                mfu_consecutive_steps=10,
            )
        )

        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                mini_wandb=wandb,
                active_node_ids=_ACTIVE_NODE_IDS,
            )
        )

        assert decision.action == ActionType.NOTIFY_HUMAN
        assert "invalid" in decision.reason.lower()
        assert decision.trigger == TriggerType.TELEMETRY_BLIND

    def test_temperature_outlier_with_nan_mfu_does_not_report_healthy(self) -> None:
        """The key regression: NaN MFU must never be interpreted as 'MFU is healthy'."""
        store = make_fake_metric_store()
        for i in range(8):
            inject_gpu_temperature(store, node_id="node-0", gpu=str(i), celsius=60.0)
            inject_gpu_temperature(store, node_id="node-1", gpu=str(i), celsius=105.0)

        wandb = make_fake_mini_wandb(steps={i: {"mfu": float("nan")} for i in range(1, 11)})
        detector = ThermalThrottlingDetector(
            config=ThermalThrottlingDetectorConfig(
                temperature_delta_threshold=20.0,
                mfu_baseline=0.5,
                mfu_consecutive_steps=10,
            )
        )

        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                mini_wandb=wandb,
                active_node_ids=_ACTIVE_NODE_IDS,
            )
        )

        assert "MFU is healthy" not in decision.reason


class TestThermalThrottlingDetectorValidation:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(temperature_delta_threshold=0.0), "temperature_delta_threshold"),
            (dict(temperature_delta_threshold=-1.0), "temperature_delta_threshold"),
            (dict(mfu_decline_threshold_ratio=0.0), "mfu_decline_threshold_ratio"),
            (dict(mfu_decline_threshold_ratio=1.5), "mfu_decline_threshold_ratio"),
            (dict(mfu_consecutive_steps=0), "mfu_consecutive_steps"),
            (dict(mfu_baseline_steps=0), "mfu_baseline_steps"),
        ],
    )
    def test_invalid_parameter_rejected(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ThermalThrottlingDetectorConfig(**kwargs)
