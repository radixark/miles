from datetime import datetime, timedelta, timezone

import pytest
from tests.fast.utils.ft.utils import inject_nic_down, inject_nic_up, make_detector_context, make_fake_metric_store

from miles.utils.ft.controller.detectors.core.network import NetworkAlertDetector, NetworkAlertDetectorConfig
from miles.utils.ft.controller.types import ActionType


class TestNetworkAlertDetector:
    def test_all_healthy(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_nic_up(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=2))
        inject_nic_up(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=1))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_single_alert_below_threshold(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_nic_down(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=2))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_two_alerts_triggers(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_nic_up(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=4))
        inject_nic_down(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=3))
        inject_nic_up(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=2))
        inject_nic_down(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=1))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                active_node_ids={"node-0"},
            )
        )

        assert decision.action == ActionType.ENTER_RECOVERY
        assert "node-0" in decision.bad_node_ids

    def test_multi_node_each_one_alert(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_nic_down(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=2))
        inject_nic_down(store, node_id="node-1", device="ib0", timestamp=now - timedelta(minutes=2))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_alerts_outside_window_ignored(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_nic_down(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=10))
        inject_nic_down(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=8))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_empty_store(self) -> None:
        store = make_fake_metric_store()
        detector = NetworkAlertDetector()
        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_permanent_nic_crash_detected(self) -> None:
        """A NIC that goes down once and stays down (permanent crash) was
        previously missed because only up→down flap transitions were counted.
        Now persistent-down detection catches this case."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        # NIC was up, then crashed permanently — only 1 down event
        inject_nic_up(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=4))
        inject_nic_down(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=3))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                active_node_ids={"node-0"},
            )
        )

        assert decision.action == ActionType.ENTER_RECOVERY
        assert "node-0" in decision.bad_node_ids

    def test_nic_down_then_recovered_no_persistent_fault(self) -> None:
        """A NIC that went down but came back up should not be flagged as
        persistent down."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_nic_up(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=4))
        inject_nic_down(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=3))
        inject_nic_up(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=2))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                active_node_ids={"node-0"},
            )
        )

        assert decision.action == ActionType.NONE

    def test_flap_and_persistent_on_different_nodes_both_detected(self) -> None:
        """Both flapping NIC (multiple transitions) and permanent crash
        (single down, stays down) are detected on different nodes."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        # node-0: flapping NIC (2 transitions)
        inject_nic_up(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=4))
        inject_nic_down(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=3))
        inject_nic_up(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=2))
        inject_nic_down(store, node_id="node-0", device="ib0", timestamp=now - timedelta(minutes=1))

        # node-1: permanent crash (1 transition, stays down)
        inject_nic_up(store, node_id="node-1", device="ib0", timestamp=now - timedelta(minutes=4))
        inject_nic_down(store, node_id="node-1", device="ib0", timestamp=now - timedelta(minutes=3))

        detector = NetworkAlertDetector()
        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                active_node_ids={"node-0", "node-1"},
            )
        )

        assert decision.action == ActionType.ENTER_RECOVERY
        assert set(decision.bad_node_ids) == {"node-0", "node-1"}


class TestNetworkAlertDetectorValidation:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(alert_window_minutes=0), "alert_window_minutes must be positive"),
            (dict(alert_window_minutes=-1), "alert_window_minutes must be positive"),
            (dict(alert_threshold=0), "alert_threshold must be >= 1"),
            (dict(alert_threshold=-1), "alert_threshold must be >= 1"),
        ],
    )
    def test_invalid_parameter_rejected(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            NetworkAlertDetectorConfig(**kwargs)
