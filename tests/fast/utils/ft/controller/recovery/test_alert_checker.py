"""Unit tests for AlertChecker."""
from datetime import timedelta

from miles.utils.ft.controller.recovery.alert_checker import AlertChecker
from miles.utils.ft.models.fault import unique_node_ids
from miles.utils.ft.models.metric_names import GPU_AVAILABLE
from miles.utils.ft.models.metrics import GaugeSample
from tests.fast.utils.ft.conftest import (
    inject_critical_xid,
    inject_gpu_unavailable,
    inject_nic_down,
    inject_nic_up,
    make_fake_metric_store,
)


class TestAlertCheckerEmpty:
    def test_no_metrics_returns_empty(self) -> None:
        store = make_fake_metric_store()
        checker = AlertChecker(metric_store=store)

        faults = checker.check_alerts()

        assert faults == []

    def test_healthy_metrics_returns_empty(self) -> None:
        store = make_fake_metric_store(metrics=[
            GaugeSample(name=GPU_AVAILABLE, labels={"gpu": "0"}, value=1.0),
        ])
        checker = AlertChecker(metric_store=store)

        faults = checker.check_alerts()

        assert faults == []


class TestAlertCheckerMultiFaultSameNode:
    def test_multiple_faults_same_node_deduplicates_node_id(self) -> None:
        store = make_fake_metric_store()
        inject_gpu_unavailable(store, node_id="node-0", gpu="0")
        inject_critical_xid(store, node_id="node-0")
        checker = AlertChecker(metric_store=store)

        faults = checker.check_alerts()

        assert sorted(unique_node_ids(faults)) == ["node-0"]
        assert len(faults) == 2

    def test_reasons_follow_check_order(self) -> None:
        store = make_fake_metric_store()
        inject_gpu_unavailable(store, node_id="node-0")
        inject_critical_xid(store, node_id="node-0")
        checker = AlertChecker(metric_store=store)

        faults = checker.check_alerts()
        reasons = [f.reason for f in faults]

        assert "GPU unavailable" in reasons[0]
        assert "non-auto-recoverable XID" in reasons[1]


class TestAlertCheckerMultiNode:
    def test_multiple_bad_nodes_sorted(self) -> None:
        store = make_fake_metric_store()
        inject_gpu_unavailable(store, node_id="node-2")
        inject_gpu_unavailable(store, node_id="node-0")
        inject_gpu_unavailable(store, node_id="node-1")
        checker = AlertChecker(metric_store=store)

        faults = checker.check_alerts()

        assert sorted(unique_node_ids(faults)) == ["node-0", "node-1", "node-2"]
        assert len(faults) == 3


class TestAlertCheckerNetworkAlerts:
    def test_no_network_data_returns_empty(self) -> None:
        store = make_fake_metric_store()
        checker = AlertChecker(metric_store=store)

        faults = checker.check_alerts()

        assert faults == []

    def test_all_nics_up_returns_empty(self) -> None:
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib1")
        checker = AlertChecker(metric_store=store)

        faults = checker.check_alerts()

        assert faults == []

    def test_consecutive_nic_down_detected(self) -> None:
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        checker = AlertChecker(metric_store=store)

        faults = checker.check_alerts()
        node_ids = [f.node_id for f in faults]

        assert "node-0" in node_ids
        assert any("NIC down" in f.reason for f in faults)

    def test_nic_flapping_faults_are_ephemeral(self) -> None:
        """NIC-down-in-window faults returned by check_alerts should be ephemeral."""
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib1")
        inject_nic_up(store, node_id="node-0", device="ib2")
        inject_nic_up(store, node_id="node-0", device="ib3")
        checker = AlertChecker(metric_store=store)

        faults = checker.check_alerts()
        nic_faults = [f for f in faults if "NIC down" in f.reason]

        assert len(nic_faults) == 1
        assert nic_faults[0].ephemeral is True

    def test_single_nic_down_below_threshold(self) -> None:
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib1")
        inject_nic_up(store, node_id="node-0", device="ib2")
        checker = AlertChecker(
            metric_store=store,
            network_alert_threshold=2,
        )

        faults = checker.check_alerts()
        node_ids = [f.node_id for f in faults]

        assert "node-0" not in node_ids

    def test_custom_threshold(self) -> None:
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        checker = AlertChecker(
            metric_store=store,
            network_alert_threshold=3,
        )

        faults = checker.check_alerts()
        node_ids = [f.node_id for f in faults]

        assert "node-0" in node_ids

    def test_network_and_hardware_faults_combined(self) -> None:
        store = make_fake_metric_store()
        inject_gpu_unavailable(store, node_id="node-1")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        checker = AlertChecker(metric_store=store)

        faults = checker.check_alerts()
        node_ids = [f.node_id for f in faults]

        assert "node-0" in node_ids
        assert "node-1" in node_ids
        assert any("GPU unavailable" in f.reason for f in faults)
        assert any("NIC down" in f.reason for f in faults)

    def test_multiple_nodes_network_alerts(self) -> None:
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-2", device="ib0")
        inject_nic_down(store, node_id="node-2", device="ib0")
        checker = AlertChecker(metric_store=store)

        faults = checker.check_alerts()

        assert sorted(unique_node_ids(faults)) == ["node-0", "node-2"]
