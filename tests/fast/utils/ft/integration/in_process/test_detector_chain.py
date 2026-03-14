"""Integration tests for the detector chain priority and short-circuit behavior."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from tests.fast.utils.ft.utils import (
    inject_gpu_unavailable,
    inject_healthy_node,
    inject_heartbeat,
    make_detector_context,
    make_fake_metric_store,
    make_fake_mini_wandb,
)

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.detectors.chain import DetectorChainConfig, build_detector_chain
from miles.utils.ft.controller.detectors.core.hang import HangDetector, HangDetectorConfig
from miles.utils.ft.controller.detectors.core.mfu_decline import MfuDeclineDetector, MfuDeclineDetectorConfig
from miles.utils.ft.controller.detectors.core.network import NetworkAlertDetector, NetworkAlertDetectorConfig
from miles.utils.ft.controller.types import ActionType
from miles.utils.ft.utils.metric_names import AGENT_HEARTBEAT, NODE_NETWORK_UP

_ACTIVE_NODE_IDS: set[str] = {"node-0", "node-1"}


class TestDetectorChainIntegration:
    def test_all_healthy_returns_none(self) -> None:
        run_id = "detector-chain-run"
        store = make_fake_metric_store()
        inject_healthy_node(store, node_id="node-0")
        inject_healthy_node(store, node_id="node-1")

        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, rank="0", timestamp=now - timedelta(minutes=5), ft_run_id=run_id)
        inject_heartbeat(store, value=110.0, rank="0", timestamp=now - timedelta(minutes=1), ft_run_id=run_id)

        wandb = make_fake_mini_wandb(steps={i: {"loss": 2.5, "mfu": 0.45} for i in range(1, 11)})
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=wandb,
            active_node_ids=_ACTIVE_NODE_IDS,
            job_status=JobStatus.RUNNING,
            active_run_id=run_id,
        )
        chain = build_detector_chain()

        for detector in chain:
            decision = detector.evaluate(ctx)
            assert decision.action == ActionType.NONE

    def test_hardware_fault_overrides_crash(self) -> None:
        """GpuFaultDetector has higher priority than TrainingCrashDetector."""
        store = make_fake_metric_store()
        inject_gpu_unavailable(store, node_id="node-0")

        wandb = make_fake_mini_wandb(steps={1: {"loss": 2.5}})
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=wandb,
            active_node_ids=_ACTIVE_NODE_IDS,
            job_status=JobStatus.FAILED,
        )
        chain = build_detector_chain()

        for detector in chain:
            decision = detector.evaluate(ctx)
            if decision.action != ActionType.NONE:
                break

        assert decision.action == ActionType.ENTER_RECOVERY
        assert "node-0" in decision.bad_node_ids
        assert "GPU unavailable" in decision.reason

    def test_network_alert_overrides_hang(self) -> None:
        """NetworkAlertDetector has higher priority than HangDetector."""
        store = make_fake_metric_store()

        now = datetime.now(timezone.utc)
        store.ingest_samples(
            target_id="node-0",
            samples=[GaugeSample(name=NODE_NETWORK_UP, labels={"device": "ib0"}, value=1.0)],
            timestamp=now - timedelta(minutes=4),
        )
        store.ingest_samples(
            target_id="node-0",
            samples=[GaugeSample(name=NODE_NETWORK_UP, labels={"device": "ib0"}, value=0.0)],
            timestamp=now - timedelta(minutes=3),
        )
        store.ingest_samples(
            target_id="node-0",
            samples=[GaugeSample(name=NODE_NETWORK_UP, labels={"device": "ib0"}, value=1.0)],
            timestamp=now - timedelta(minutes=2),
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
            active_node_ids=_ACTIVE_NODE_IDS,
            job_status=JobStatus.RUNNING,
            seconds_since_run_start=0.0,
        )
        chain = build_detector_chain()

        for detector in chain:
            decision = detector.evaluate(ctx)
            if decision.action != ActionType.NONE:
                break

        assert decision.action == ActionType.ENTER_RECOVERY
        assert "NIC" in decision.reason

    def test_crash_with_nan_loss(self) -> None:
        """TrainingCrashDetector sets trigger to nan_loss when last loss is NaN."""
        store = make_fake_metric_store()

        wandb = make_fake_mini_wandb(steps={1: {"loss": float("nan")}})
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=wandb,
            active_node_ids=_ACTIVE_NODE_IDS,
            job_status=JobStatus.FAILED,
            seconds_since_run_start=0.0,
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
            "GpuFaultDetector",
            "NicMajorityDownDetector",
            "DiskSpaceLowDetector",
            "ThermalThrottlingDetector",
            "CollectorHealthDetector",
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
        chain = build_detector_chain(
            config=DetectorChainConfig(
                hang=HangDetectorConfig(training_timeout_minutes=20),
            )
        )
        hang = next(d for d in chain if isinstance(d, HangDetector))
        assert hang._config.training_timeout_minutes == 20

    def test_mfu_threshold_ratio(self) -> None:
        chain = build_detector_chain(
            config=DetectorChainConfig(
                mfu=MfuDeclineDetectorConfig(mfu_threshold_ratio=0.5),
            )
        )
        mfu = next(d for d in chain if isinstance(d, MfuDeclineDetector))
        assert mfu._config.mfu_threshold_ratio == 0.5

    def test_network_alert_window_minutes(self) -> None:
        chain = build_detector_chain(
            config=DetectorChainConfig(
                network=NetworkAlertDetectorConfig(alert_window_minutes=10),
            )
        )
        net = next(d for d in chain if isinstance(d, NetworkAlertDetector))
        assert net._alert_window == timedelta(minutes=10)

    def test_network_alert_threshold(self) -> None:
        chain = build_detector_chain(
            config=DetectorChainConfig(
                network=NetworkAlertDetectorConfig(alert_threshold=5),
            )
        )
        net = next(d for d in chain if isinstance(d, NetworkAlertDetector))
        assert net._alert_threshold == 5

    def test_multiple_config_keys(self) -> None:
        chain = build_detector_chain(
            config=DetectorChainConfig(
                hang=HangDetectorConfig(training_timeout_minutes=30),
                mfu=MfuDeclineDetectorConfig(mfu_threshold_ratio=0.6),
                network=NetworkAlertDetectorConfig(alert_window_minutes=15, alert_threshold=3),
            )
        )

        hang = next(d for d in chain if isinstance(d, HangDetector))
        mfu = next(d for d in chain if isinstance(d, MfuDeclineDetector))
        net = next(d for d in chain if isinstance(d, NetworkAlertDetector))

        assert hang._config.training_timeout_minutes == 30
        assert mfu._config.mfu_threshold_ratio == 0.6
        assert net._alert_window == timedelta(minutes=15)
        assert net._alert_threshold == 3

    def test_unknown_config_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DetectorChainConfig(**{"unknown_key": 42})  # type: ignore[arg-type]
