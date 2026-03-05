from tests.fast.utils.ft.helpers import (
    inject_critical_xid,
    inject_disk_fault,
    inject_gpu_unavailable,
    inject_healthy_node,
    inject_nic_down,
    inject_nic_up,
    make_detector_context,
    make_fake_metric_store,
)

from miles.utils.ft.controller.detectors.hardware import HighConfidenceHardwareDetector
from miles.utils.ft.models import ActionType


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

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-0" in decision.bad_node_ids
        assert "GPU unavailable" in decision.reason

    def test_critical_xid_48(self) -> None:
        store = make_fake_metric_store()
        inject_critical_xid(store, node_id="node-0", xid_code=48)
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-0" in decision.bad_node_ids
        assert "XID 48" in decision.reason

    def test_critical_xid_62(self) -> None:
        store = make_fake_metric_store()
        inject_critical_xid(store, node_id="node-0", xid_code=62)
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-0" in decision.bad_node_ids
        assert "XID 62" in decision.reason

    def test_critical_xid_64(self) -> None:
        store = make_fake_metric_store()
        inject_critical_xid(store, node_id="node-0", xid_code=64)
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-0" in decision.bad_node_ids
        assert "XID 64" in decision.reason

    def test_critical_xid_79(self) -> None:
        store = make_fake_metric_store()
        inject_critical_xid(store, node_id="node-0", xid_code=79)
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-0" in decision.bad_node_ids
        assert "XID 79" in decision.reason

    def test_non_critical_xid_ignored(self) -> None:
        store = make_fake_metric_store()
        inject_critical_xid(store, node_id="node-0", xid_code=31)
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_disk_space_low(self) -> None:
        store = make_fake_metric_store()
        inject_disk_fault(store, node_id="node-0", available_bytes=500e6)  # 500MB < 1GB
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-0" in decision.bad_node_ids
        assert "disk space low" in decision.reason

    def test_disk_space_above_threshold(self) -> None:
        store = make_fake_metric_store()
        inject_disk_fault(store, node_id="node-0", available_bytes=2e9)  # 2GB > 1GB
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_majority_nic_down(self) -> None:
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib1")
        inject_nic_down(store, node_id="node-0", device="ib2")
        inject_nic_up(store, node_id="node-0", device="ib3")
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
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
        inject_critical_xid(store, node_id="node-1", xid_code=48)
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-0" in decision.bad_node_ids
        assert "node-1" in decision.bad_node_ids

    def test_empty_metric_store(self) -> None:
        store = make_fake_metric_store()
        detector = HighConfidenceHardwareDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE
