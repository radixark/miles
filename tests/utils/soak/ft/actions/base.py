from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.types import BaseSoakActionForm, SoakActionDetails, SoakActionRequest
from tests.utils.soak.core.views import SoakActionRecord, is_normal_step, trainer_step_ends
from tests.utils.soak.ft.recovery import compute_recovered_at
from tests.utils.soak.ft.types import CellTarget

CellFaultForms = dict[str, list["BaseCellFaultForm"]]


class BaseCellFaultForm(BaseSoakActionForm):
    @property
    def harms_target(self) -> bool:
        return True

    @property
    def needs_fault_target(self) -> bool:
        return False

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
