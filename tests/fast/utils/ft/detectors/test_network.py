from datetime import datetime, timedelta, timezone

from tests.fast.utils.ft.conftest import (
    EMPTY_RANK_PLACEMENT,
    make_fake_metric_store,
    make_fake_mini_wandb,
)

from miles.utils.ft.controller.detectors._metric_names import NODE_NIC_UP
from miles.utils.ft.controller.detectors.network import NetworkAlertDetector
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus
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
        samples=[MetricSample(name=NODE_NIC_UP, labels={"device": device}, value=value)],
        timestamp=timestamp,
    )


class TestNetworkAlertDetector:
    def test_all_healthy(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        _inject_nic_at_time(store, "node-0", "ib0", 1.0, now - timedelta(minutes=2))
        _inject_nic_at_time(store, "node-0", "ib0", 1.0, now - timedelta(minutes=1))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.NONE

    def test_single_alert_below_threshold(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        _inject_nic_at_time(store, "node-0", "ib0", 0.0, now - timedelta(minutes=2))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.NONE

    def test_two_alerts_triggers(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        _inject_nic_at_time(store, "node-0", "ib0", 0.0, now - timedelta(minutes=3))
        _inject_nic_at_time(store, "node-0", "ib0", 0.0, now - timedelta(minutes=1))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-0" in decision.bad_node_ids

    def test_multi_node_each_one_alert(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        _inject_nic_at_time(store, "node-0", "ib0", 0.0, now - timedelta(minutes=2))
        _inject_nic_at_time(store, "node-1", "ib0", 0.0, now - timedelta(minutes=2))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.NONE

    def test_alerts_outside_window_ignored(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        _inject_nic_at_time(store, "node-0", "ib0", 0.0, now - timedelta(minutes=10))
        _inject_nic_at_time(store, "node-0", "ib0", 0.0, now - timedelta(minutes=8))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.NONE

    def test_empty_store(self) -> None:
        store = make_fake_metric_store()
        detector = NetworkAlertDetector()
        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.NONE
