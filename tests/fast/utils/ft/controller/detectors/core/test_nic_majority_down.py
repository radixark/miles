from tests.fast.utils.ft.utils import inject_nic_down, inject_nic_up, make_detector_context, make_fake_metric_store

from miles.utils.ft.controller.detectors.core.nic_majority_down import NicMajorityDownDetector
from miles.utils.ft.controller.types import ActionType


class TestNicMajorityDownDetector:
    def test_majority_nic_down(self) -> None:
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib1")
        inject_nic_down(store, node_id="node-0", device="ib2")
        inject_nic_up(store, node_id="node-0", device="ib3")
        detector = NicMajorityDownDetector()

        decision = detector.evaluate(
            make_detector_context(
                metric_store=store,
                active_node_ids={"node-0"},
            )
        )

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
        detector = NicMajorityDownDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_single_nic_down_ignored(self) -> None:
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib1")
        inject_nic_up(store, node_id="node-0", device="ib2")
        inject_nic_up(store, node_id="node-0", device="ib3")
        detector = NicMajorityDownDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_empty_metric_store_returns_telemetry_blind(self) -> None:
        store = make_fake_metric_store()
        detector = NicMajorityDownDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NOTIFY_HUMAN
