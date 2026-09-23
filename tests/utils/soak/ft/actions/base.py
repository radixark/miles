from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.types import BaseSoakActionForm, SoakActionDetails, SoakActionRequest
from tests.utils.soak.core.views import SoakActionRecord, is_normal_step, trainer_step_ends
from tests.utils.soak.ft.recovery import compute_recovered_at
from tests.utils.soak.ft.types import CellTarget

from miles.utils.test_utils.fault_injector.models import ObservedFaultHookTarget

CellFaultForms = dict[str, list["BaseCellFaultForm"]]


class BaseCellFaultForm(BaseSoakActionForm):
    @property
    def harms_target(self) -> bool:
        return True

    @property
    def needs_fault_target(self) -> bool:
        return False

    @property
    def trigger_cell_types(self) -> frozenset[str]:
        return frozenset()

    @property
    def process_patterns(self) -> dict[str, str]:
        return {}

    def is_recovered(self, *, action: SoakActionRecord, events: list[SoakEvent]) -> bool:
        if action.applied is None or action.result is None or not action.result.returned:
            return False

        recovered_at = compute_recovered_at(action=action, events=events)
        if recovered_at is None:
            return False

        return any(step.timestamp > recovered_at and is_normal_step(step) for step in trainer_step_ends(events))

    def _create_request(self, *, target: CellTarget, details: SoakActionDetails) -> SoakActionRequest:
        return SoakActionRequest(target=target, form_name=self.name, details=details)


def resolve_fault_target(target: CellTarget) -> ObservedFaultHookTarget | None:
    if (fault_target := target.fault_target) is None or fault_target.workers_hash != target.incarnation:
        return None
    return fault_target


def assert_request_target(
    request: SoakActionRequest, *, fault_target: ObservedFaultHookTarget
) -> ObservedFaultHookTarget:
    assert fault_target.cell_id == request.target.identity
    assert fault_target.workers_hash == request.target.incarnation
    return fault_target
