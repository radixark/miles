import pytest
from tests.fast.utils.ft.utils.metric_injectors import make_detector_context

from miles.utils.ft.controller.detectors.base import (
    BaseFaultDetector,
    DetectorContext,
    _filter_node_ids_by_active,
)
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType


class TestBaseFaultDetector:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseFaultDetector()  # type: ignore[abstract]

    def test_subclass_must_implement_evaluate(self) -> None:
        class _IncompleteDetector(BaseFaultDetector):
            pass

        with pytest.raises(TypeError):
            _IncompleteDetector()  # type: ignore[abstract]

    def test_subclass_with_evaluate_raw_can_be_instantiated(self) -> None:
        class _CompleteDetector(BaseFaultDetector):
            def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
                return Decision(action=ActionType.NONE, reason="test")

        detector = _CompleteDetector()
        assert isinstance(detector, BaseFaultDetector)


class TestFilterNodeIdsByActive:
    def test_keeps_only_active_nodes(self) -> None:
        result = _filter_node_ids_by_active(["n1", "n2", "n3"], active_node_ids={"n1", "n3"})
        assert result == ["n1", "n3"]

    def test_preserves_original_order(self) -> None:
        result = _filter_node_ids_by_active(["n3", "n1", "n2"], active_node_ids={"n1", "n2", "n3"})
        assert result == ["n3", "n1", "n2"]

    def test_no_intersection_returns_empty(self) -> None:
        result = _filter_node_ids_by_active(["n1", "n2"], active_node_ids={"n3", "n4"})
        assert result == []

    def test_empty_inputs_return_empty(self) -> None:
        assert _filter_node_ids_by_active([], active_node_ids=set()) == []
        assert _filter_node_ids_by_active([], active_node_ids={"n1"}) == []
        assert _filter_node_ids_by_active(["n1"], active_node_ids=set()) == []


class _StubDetector(BaseFaultDetector):
    def __init__(self, decision: Decision) -> None:
        self._decision = decision

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        return self._decision


class TestBaseFaultDetectorEvaluate:
    def test_bad_nodes_all_active_returned_unchanged(self) -> None:
        decision = Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=["n1", "n2"],
            reason="fault",
            trigger=TriggerType.HARDWARE,
        )
        detector = _StubDetector(decision=decision)
        ctx = make_detector_context(active_node_ids={"n1", "n2"})

        result = detector.evaluate(ctx)
        assert result.bad_node_ids == ["n1", "n2"]
        assert result.action == ActionType.ENTER_RECOVERY

    def test_bad_nodes_partially_active_are_filtered(self) -> None:
        decision = Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=["n1", "n2", "n3"],
            reason="fault",
            trigger=TriggerType.HARDWARE,
        )
        detector = _StubDetector(decision=decision)
        ctx = make_detector_context(active_node_ids={"n1", "n3"})

        result = detector.evaluate(ctx)
        assert result.bad_node_ids == ["n1", "n3"]

    def test_bad_nodes_all_inactive_returns_no_fault(self) -> None:
        decision = Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=["n1", "n2"],
            reason="fault",
            trigger=TriggerType.HARDWARE,
        )
        detector = _StubDetector(decision=decision)
        ctx = make_detector_context(active_node_ids={"n3", "n4"})

        result = detector.evaluate(ctx)
        assert result.action == ActionType.NONE
        assert result.bad_node_ids == []

    def test_no_bad_nodes_returned_unchanged(self) -> None:
        decision = Decision.no_fault(reason="all clear")
        detector = _StubDetector(decision=decision)
        ctx = make_detector_context(active_node_ids={"n1"})

        result = detector.evaluate(ctx)
        assert result.action == ActionType.NONE
        assert result.reason == "all clear"

    def test_empty_active_node_ids_with_bad_nodes_returns_no_fault(self) -> None:
        """When active_node_ids is empty, bad_node_ids should not pass through
        unfiltered. Previously the filter condition `if bad_node_ids and active_node_ids`
        was False when active_node_ids was empty, returning the raw decision."""
        decision = Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=["n1", "n2"],
            reason="fault",
            trigger=TriggerType.HARDWARE,
        )
        detector = _StubDetector(decision=decision)
        ctx = make_detector_context(active_node_ids=set())

        result = detector.evaluate(ctx)
        assert result.action == ActionType.NONE
        assert "no active nodes" in result.reason

    def test_empty_active_node_ids_without_bad_nodes_returns_no_fault(self) -> None:
        """Even a no-fault decision should return no_fault when no active nodes."""
        decision = Decision.no_fault(reason="all clear")
        detector = _StubDetector(decision=decision)
        ctx = make_detector_context(active_node_ids=set())

        result = detector.evaluate(ctx)
        assert result.action == ActionType.NONE
        assert "no active nodes" in result.reason
