"""Unit tests for AlertChecker."""
from miles.utils.ft.controller.recovery_orchestrator.alert_checker import AlertChecker
from miles.utils.ft.metric_names import GPU_AVAILABLE, XID_CODE_RECENT
from miles.utils.ft.models import MetricSample
from tests.fast.utils.ft.helpers import (
    inject_critical_xid,
    inject_gpu_unavailable,
    make_fake_metric_store,
)


class TestAlertCheckerEmpty:
    def test_no_metrics_returns_empty(self) -> None:
        store = make_fake_metric_store()
        checker = AlertChecker(metric_store=store)

        bad_node_ids, reasons = checker.check_alerts()

        assert bad_node_ids == []
        assert reasons == []

    def test_healthy_metrics_returns_empty(self) -> None:
        store = make_fake_metric_store(metrics=[
            MetricSample(name=GPU_AVAILABLE, labels={"gpu": "0"}, value=1.0),
        ])
        checker = AlertChecker(metric_store=store)

        bad_node_ids, reasons = checker.check_alerts()

        assert bad_node_ids == []
        assert reasons == []


class TestAlertCheckerMultiFaultSameNode:
    def test_multiple_faults_same_node_deduplicates_node_id(self) -> None:
        store = make_fake_metric_store()
        inject_gpu_unavailable(store, node_id="node-0", gpu="0")
        inject_critical_xid(store, node_id="node-0", xid_code=48)
        checker = AlertChecker(metric_store=store)

        bad_node_ids, reasons = checker.check_alerts()

        assert bad_node_ids == ["node-0"]
        assert len(reasons) == 2

    def test_reasons_follow_check_order(self) -> None:
        store = make_fake_metric_store()
        inject_gpu_unavailable(store, node_id="node-0")
        inject_critical_xid(store, node_id="node-0", xid_code=64)
        checker = AlertChecker(metric_store=store)

        _, reasons = checker.check_alerts()

        assert "GPU unavailable" in reasons[0]
        assert "critical XID 64" in reasons[1]


class TestAlertCheckerMultiNode:
    def test_multiple_bad_nodes_sorted(self) -> None:
        store = make_fake_metric_store()
        inject_gpu_unavailable(store, node_id="node-2")
        inject_gpu_unavailable(store, node_id="node-0")
        inject_gpu_unavailable(store, node_id="node-1")
        checker = AlertChecker(metric_store=store)

        bad_node_ids, reasons = checker.check_alerts()

        assert bad_node_ids == ["node-0", "node-1", "node-2"]
        assert len(reasons) == 3
