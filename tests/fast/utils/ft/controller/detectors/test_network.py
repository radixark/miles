from datetime import datetime, timedelta, timezone

import pytest
from tests.fast.utils.ft.helpers import make_detector_context, make_fake_metric_store

from miles.utils.ft.controller.detectors.network import NetworkAlertDetector
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus
from miles.utils.ft.metric_names import NODE_NETWORK_UP
from miles.utils.ft.models import ActionType, MetricSample


def _inject_nic_at_time(
    store: MiniPrometheus,
    node_id: str,
    device: str,
    value: float,
    timestamp: datetime,
) -> None:
    store.ingest_samples(
        target_id=node_id,
        samples=[MetricSample(name=NODE_NETWORK_UP, labels={"device": device}, value=value)],
        timestamp=timestamp,
    )


class TestNetworkAlertDetector:
    def test_all_healthy(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        _inject_nic_at_time(store, "node-0", "ib0", 1.0, now - timedelta(minutes=2))
        _inject_nic_at_time(store, "node-0", "ib0", 1.0, now - timedelta(minutes=1))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_single_alert_below_threshold(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        _inject_nic_at_time(store, "node-0", "ib0", 0.0, now - timedelta(minutes=2))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_two_alerts_triggers(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        _inject_nic_at_time(store, "node-0", "ib0", 0.0, now - timedelta(minutes=3))
        _inject_nic_at_time(store, "node-0", "ib0", 0.0, now - timedelta(minutes=1))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-0" in decision.bad_node_ids

    def test_multi_node_each_one_alert(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        _inject_nic_at_time(store, "node-0", "ib0", 0.0, now - timedelta(minutes=2))
        _inject_nic_at_time(store, "node-1", "ib0", 0.0, now - timedelta(minutes=2))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_alerts_outside_window_ignored(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        _inject_nic_at_time(store, "node-0", "ib0", 0.0, now - timedelta(minutes=10))
        _inject_nic_at_time(store, "node-0", "ib0", 0.0, now - timedelta(minutes=8))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_empty_store(self) -> None:
        store = make_fake_metric_store()
        detector = NetworkAlertDetector()
        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE


class TestNetworkAlertDetectorValidation:
    def test_zero_alert_window_rejected(self) -> None:
        with pytest.raises(ValueError, match="alert_window"):
            NetworkAlertDetector(alert_window=timedelta(0))

    def test_negative_alert_window_rejected(self) -> None:
        with pytest.raises(ValueError, match="alert_window"):
            NetworkAlertDetector(alert_window=timedelta(minutes=-1))

    def test_zero_alert_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="alert_threshold"):
            NetworkAlertDetector(alert_threshold=0)

    def test_negative_alert_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="alert_threshold"):
            NetworkAlertDetector(alert_threshold=-1)
