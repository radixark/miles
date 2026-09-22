import pytest
from tests.utils.soak.core.events import SoakActionAppliedEvent, SoakActionRequestedEvent, SoakEvent
from tests.utils.soak.core.types import SoakActionRequest
from tests.utils.soak.ft.checkers.healing import MIN_SOAK_INJECTIONS, assert_min_injections
from tests.utils.soak.ft.types import ROLLOUT_CELL_TYPE, CellTarget, InjectFaultDetails, ObservedCellFault

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import FaultTarget


def _applied_injections(count: int) -> list[SoakEvent]:
    events: list[SoakEvent] = []
    for index in range(count):
        fault_target = FaultTarget(cell_id=f"rollout-{index}", sub_index=0, workers_hash="hash")
        request = SoakActionRequest(
            target=CellTarget(kind="rollout", identity=f"rollout-{index}", incarnation="hash", alive=True, ready=True),
            form_name="inject_fault:sigkill",
            details=InjectFaultDetails(fault_target=fault_target),
        )
        events.append(SoakActionRequestedEvent(request=request))
        events.append(
            SoakActionAppliedEvent(
                request_id=request.request_id,
                evidence=ObservedCellFault(
                    request_id=request.request_id,
                    target=fault_target,
                    mode=FailureMode.SIGKILL,
                    observed="replaced",
                ),
            )
        )
    return events


class TestAssertMinInjections:
    def test_minimum_injection_failure_names_the_caller_context(self) -> None:
        """The standalone check reports which soak (e.g. a rollout-only one) fell short, not just that one did."""
        with pytest.raises(AssertionError, match="rollout-only soak on engine cells"):
            assert_min_injections(
                _applied_injections(1), kind=ROLLOUT_CELL_TYPE, context="rollout-only soak on engine cells"
            )

    def test_reaching_the_minimum_injection_count_passes(self) -> None:
        """Exactly the required number of successful injections satisfies the standalone check."""
        assert_min_injections(
            _applied_injections(MIN_SOAK_INJECTIONS),
            kind=ROLLOUT_CELL_TYPE,
            context="rollout-only soak on engine cells",
        )
