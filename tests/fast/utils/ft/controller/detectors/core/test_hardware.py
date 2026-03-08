from tests.fast.utils.ft.utils import (
    inject_critical_xid,
    inject_gpu_unavailable,
    inject_healthy_node,
    inject_nic_down,
    inject_nic_up,
    make_detector_context,
    make_fake_metric_store,
)

from miles.utils.ft.controller.detectors.core.hardware import HighConfidenceHardwareDetector
from miles.utils.ft.models.fault import ActionType


class TestHighConfidenceHardwareDetector:
    def test_all_healthy(self) -> None:
        store = make_fake_metric_store()
        inject_healthy_node(store, node_id="node-0")
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_gpu_unavailable(self) -> None:
        store = make_fake_metric_store()
        inject_healthy_node(store, node_id="node-0")
        inject_gpu_unavailable(store, node_id="node-0", gpu="3")
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.ENTER_RECOVERY
        assert "node-0" in decision.bad_node_ids
        assert "GPU unavailable" in decision.reason

    def test_non_auto_recoverable_xid_triggers_mark_bad(self) -> None:
        store = make_fake_metric_store()
        inject_critical_xid(store, node_id="node-0")
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.ENTER_RECOVERY
        assert "node-0" in decision.bad_node_ids
        assert "non-auto-recoverable XID" in decision.reason

    def test_majority_nic_down(self) -> None:
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib1")
        inject_nic_down(store, node_id="node-0", device="ib2")
        inject_nic_up(store, node_id="node-0", device="ib3")
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.ENTER_RECOVERY
        assert "node-0" in decision.bad_node_ids
        assert "majority NIC down" in decision.reason

    def test_half_nic_down_ignored(self) -> None:
        """Exactly 50% NICs down does not trigger (strict majority required)."""
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib1")
        inject_nic_up(store, node_id="node-0", device="ib2")
        inject_nic_up(store, node_id="node-0", device="ib3")
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_single_nic_down_ignored(self) -> None:
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib1")
        inject_nic_up(store, node_id="node-0", device="ib2")
        inject_nic_up(store, node_id="node-0", device="ib3")
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_multi_node_faults(self) -> None:
        store = make_fake_metric_store()
        inject_gpu_unavailable(store, node_id="node-0", gpu="0")
        inject_critical_xid(store, node_id="node-1")
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.ENTER_RECOVERY
        assert "node-0" in decision.bad_node_ids
        assert "node-1" in decision.bad_node_ids

    def test_empty_metric_store(self) -> None:
        store = make_fake_metric_store()
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE
