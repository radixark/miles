from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from tests.utils.soak.core.events import SoakObservationEvent
from tests.utils.soak.core.types import SoakObserver
from tests.utils.soak.core.utils import recording_error
from tests.utils.soak.ft.actions.base import CellFaultForms
from tests.utils.soak.ft.cells import cell_is_alive, cell_is_ready, cell_type_of
from tests.utils.soak.ft.types import CellTarget

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.ft_utils.api_server.models import Cell, CellList


@dataclass(frozen=True, kw_only=True)
class CellObserver(SoakObserver):
    base_url: str
    cell_types: set[str]
    namespace: str | None = None
    release: str | None = None
    fault_target_cell_types: frozenset[str] = frozenset()
    process_patterns_of_type: dict[str, dict[str, str]] = field(default_factory=dict)

    async def observe(self) -> SoakObservationEvent:
        observed_at = datetime.now(timezone.utc)
        errors: dict[str, str] = {}

        cells = await self._observe_cells(errors=errors)

        return SoakObservationEvent(
            timestamp=observed_at,
            targets=None if cells is None else [_create_cell_target(cell) for cell in cells],
            errors=errors,
        )

    async def _observe_cells(self, *, errors: dict[str, str]) -> list[Cell] | None:
        cells: list[Cell] | None = None
        with recording_error(errors, "cells"):
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/v1/cells")
                response.raise_for_status()
                cells = [
                    cell
                    for cell in CellList.model_validate(response.json()).items
                    if cell_type_of(cell) in self.cell_types
                ]
        return cells


def create_cell_observer(
    *,
    base_url: str,
    cell_types: set[str],
    forms: CellFaultForms,
    config: ExecuteTrainConfig,
) -> CellObserver:
    return CellObserver(base_url=base_url, cell_types=cell_types)


def _create_cell_target(cell: Cell) -> CellTarget:
    return CellTarget(
        kind=cell_type_of(cell),
        identity=cell.metadata.name,
        incarnation=cell.status.workers_hash,
        alive=cell_is_alive(cell),
        ready=cell_is_ready(cell),
    )
