from dataclasses import dataclass, field

from tests.utils.soak.core.events import SoakObservationEvent
from tests.utils.soak.core.types import SoakObserver
from tests.utils.soak.ft.actions.base import CellFaultForms

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig


@dataclass(frozen=True, kw_only=True)
class CellObserver(SoakObserver):
    base_url: str
    cell_types: set[str]
    namespace: str | None = None
    release: str | None = None
    fault_target_cell_types: frozenset[str] = frozenset()
    process_patterns_of_type: dict[str, dict[str, str]] = field(default_factory=dict)

    async def observe(self) -> SoakObservationEvent:
        raise NotImplementedError


def create_cell_observer(
    *,
    base_url: str,
    cell_types: set[str],
    forms: CellFaultForms,
    config: ExecuteTrainConfig,
) -> CellObserver:
    raise NotImplementedError
