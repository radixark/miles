import pytest

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.models import ActionType, Decision


class TestBaseFaultDetector:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseFaultDetector()  # type: ignore[abstract]

    def test_subclass_must_implement_evaluate(self) -> None:
        class _IncompleteDetector(BaseFaultDetector):
            pass

        with pytest.raises(TypeError):
            _IncompleteDetector()  # type: ignore[abstract]

    def test_subclass_with_evaluate_can_be_instantiated(self) -> None:
        class _CompleteDetector(BaseFaultDetector):
            def evaluate(self, ctx: DetectorContext) -> Decision:
                return Decision(action=ActionType.NONE, reason="test")

        detector = _CompleteDetector()
        assert isinstance(detector, BaseFaultDetector)
