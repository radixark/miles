"""Integration tests for the detector chain priority and short-circuit behavior."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from tests.fast.utils.ft.helpers import (
    inject_gpu_unavailable,
    inject_healthy_node,
    make_detector_context,
    make_fake_metric_store,
    make_fake_mini_wandb,
)

from miles.utils.ft.controller.detectors.chain import DetectorChainConfig, build_detector_chain
from miles.utils.ft.controller.detectors.hang import HangDetector, HangDetectorConfig
from miles.utils.ft.controller.detectors.mfu_decline import MfuDeclineDetector, MfuDeclineDetectorConfig
from miles.utils.ft.controller.detectors.network import NetworkAlertDetector, NetworkAlertDetectorConfig
from miles.utils.ft.models.metric_names import AGENT_HEARTBEAT, NODE_NETWORK_UP
from miles.utils.ft.models.fault import ActionType
from miles.utils.ft.models.metrics import GaugeSample
from miles.utils.ft.protocols.platform import JobStatus

_RANK_PLACEMENT: dict[int, str] = {0: "node-0", 1: "node-1"}


class TestDetectorChainIntegration:
    def test_all_healthy_returns_none(self) -> None:
        store = make_fake_metric_store()
        inject_healthy_node(store, node_id="node-0")
        inject_healthy_node(store, node_id="node-1")

        now = datetime.now(timezone.utc)
        store.ingest_samples(
            target_id="rank-0",
            samples=[GaugeSample(name=AGENT_HEARTBEAT, labels={"rank": "0"}, value=100.0)],
            timestamp=now - timedelta(minutes=5),
        )
        store.ingest_samples(
            target_id="rank-0",
            samples=[GaugeSample(name=AGENT_HEARTBEAT, labels={"rank": "0"}, value=110.0)],
            timestamp=now - timedelta(minutes=1),
        )

        wandb = make_fake_mini_wandb(steps={i: {"loss": 2.5, "mfu": 0.45} for i in range(1, 11)})
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=wandb,
            rank_placement=_RANK_PLACEMENT,
            job_status=JobStatus.RUNNING,
        )
        chain = build_detector_chain()

        for detector in chain:
            decision = detector.evaluate(ctx)
            assert decision.action == ActionType.NONE

    def test_hardware_fault_overrides_crash(self) -> None:
        """HighConfidenceHardwareDetector has higher priority than TrainingCrashDetector."""
        store = make_fake_metric_store()
        inject_gpu_unavailable(store, node_id="node-0")

        wandb = make_fake_mini_wandb(steps={1: {"loss": 2.5}})
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=wandb,
            rank_placement=_RANK_PLACEMENT,
            job_status=JobStatus.FAILED,
        )
        chain = build_detector_chain()

        for detector in chain:
            decision = detector.evaluate(ctx)
            if decision.action != ActionType.NONE:
                break

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-0" in decision.bad_node_ids
        assert "GPU unavailable" in decision.reason

    def test_network_alert_overrides_hang(self) -> None:
        """NetworkAlertDetector has higher priority than HangDetector."""
        store = make_fake_metric_store()

        now = datetime.now(timezone.utc)
        store.ingest_samples(
            target_id="node-0",
            samples=[GaugeSample(name=NODE_NETWORK_UP, labels={"device": "ib0"}, value=0.0)],
            timestamp=now - timedelta(minutes=3),
        )
        store.ingest_samples(
            target_id="node-0",
            samples=[GaugeSample(name=NODE_NETWORK_UP, labels={"device": "ib0"}, value=0.0)],
            timestamp=now - timedelta(minutes=1),
        )

        store.ingest_samples(
            target_id="rank-0",
            samples=[GaugeSample(name=AGENT_HEARTBEAT, labels={"rank": "0"}, value=100.0)],
            timestamp=now - timedelta(minutes=5),
        )
        store.ingest_samples(
            target_id="rank-0",
            samples=[GaugeSample(name=AGENT_HEARTBEAT, labels={"rank": "0"}, value=100.0)],
            timestamp=now - timedelta(minutes=1),
        )

        wandb = make_fake_mini_wandb()
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=wandb,
            rank_placement=_RANK_PLACEMENT,
            job_status=JobStatus.RUNNING,
        )
        chain = build_detector_chain()

        for detector in chain:
            decision = detector.evaluate(ctx)
            if decision.action != ActionType.NONE:
                break

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "NIC down" in decision.reason

    def test_crash_with_nan_loss(self) -> None:
        """TrainingCrashDetector sets trigger to nan_loss when last loss is NaN."""
        store = make_fake_metric_store()

        wandb = make_fake_mini_wandb(steps={1: {"loss": float("nan")}})
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=wandb,
            rank_placement=_RANK_PLACEMENT,
            job_status=JobStatus.FAILED,
        )
        chain = build_detector_chain()

        for detector in chain:
            decision = detector.evaluate(ctx)
            if decision.action != ActionType.NONE:
                break

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "nan_loss"

    def test_priority_order(self) -> None:
        """Verify all detectors are present in the expected priority order."""
        chain = build_detector_chain()
        names = [type(d).__name__ for d in chain]

        assert names == [
            "HighConfidenceHardwareDetector",
            "NetworkAlertDetector",
            "TrainingCrashDetector",
            "HangDetector",
            "NanLossDetector",
            "MfuDeclineDetector",
        ]


class TestBuildDetectorChainConfig:
    def test_default_config_uses_defaults(self) -> None:
        chain = build_detector_chain()
        hang = next(d for d in chain if isinstance(d, HangDetector))
        assert hang._config.training_timeout_minutes == 10

    def test_none_config_uses_defaults(self) -> None:
        chain = build_detector_chain(config=None)
        hang = next(d for d in chain if isinstance(d, HangDetector))
        assert hang._config.training_timeout_minutes == 10

    def test_hang_timeout_minutes(self) -> None:
        chain = build_detector_chain(config=DetectorChainConfig(
            hang=HangDetectorConfig(training_timeout_minutes=20),
        ))
        hang = next(d for d in chain if isinstance(d, HangDetector))
        assert hang._config.training_timeout_minutes == 20

    def test_mfu_threshold_ratio(self) -> None:
        chain = build_detector_chain(config=DetectorChainConfig(
            mfu=MfuDeclineDetectorConfig(mfu_threshold_ratio=0.5),
        ))
        mfu = next(d for d in chain if isinstance(d, MfuDeclineDetector))
        assert mfu._mfu_threshold_ratio == 0.5

    def test_network_alert_window_minutes(self) -> None:
        chain = build_detector_chain(config=DetectorChainConfig(
            network=NetworkAlertDetectorConfig(alert_window_minutes=10),
        ))
        net = next(d for d in chain if isinstance(d, NetworkAlertDetector))
        assert net._alert_window == timedelta(minutes=10)

    def test_network_alert_threshold(self) -> None:
        chain = build_detector_chain(config=DetectorChainConfig(
            network=NetworkAlertDetectorConfig(alert_threshold=5),
        ))
        net = next(d for d in chain if isinstance(d, NetworkAlertDetector))
        assert net._alert_threshold == 5

    def test_multiple_config_keys(self) -> None:
        chain = build_detector_chain(config=DetectorChainConfig(
            hang=HangDetectorConfig(training_timeout_minutes=30),
            mfu=MfuDeclineDetectorConfig(mfu_threshold_ratio=0.6),
            network=NetworkAlertDetectorConfig(alert_window_minutes=15, alert_threshold=3),
        ))

        hang = next(d for d in chain if isinstance(d, HangDetector))
        mfu = next(d for d in chain if isinstance(d, MfuDeclineDetector))
        net = next(d for d in chain if isinstance(d, NetworkAlertDetector))

        assert hang._config.training_timeout_minutes == 30
        assert mfu._mfu_threshold_ratio == 0.6
        assert net._alert_window == timedelta(minutes=15)
        assert net._alert_threshold == 3

    def test_unknown_config_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DetectorChainConfig(**{"unknown_key": 42})  # type: ignore[arg-type]
