from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.types import BaseSoakActionForm, SoakActionDetails, SoakActionRequest
from tests.utils.soak.core.views import SoakActionRecord
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
        raise NotImplementedError

    def _create_request(self, *, target: CellTarget, details: SoakActionDetails) -> SoakActionRequest:
        return SoakActionRequest(target=target, form_name=self.name, details=details)
