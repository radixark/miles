# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import enum
import threading
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel


def cell_is_alive(cell: dict) -> bool:
    return any(cond["type"] == "Healthy" and cond["status"] == "True" for cond in cell["status"]["conditions"])


class ObservedCellState(enum.Enum):
    SUSPENDED = "Suspended"  # torn down, holding no gpu
    PENDING = "Pending"  # allocated but gated: no engine serving yet
    RUNNING_NOT_SERVING = "RunningNotServing"  # engine is up but not registered in the router
    SERVING = "Serving"  # registered in the router, i.e. actually able to answer requests


def compute_observed_cell_state(cell: dict) -> ObservedCellState:
    phase = cell["status"]["phase"]
    if phase == "Suspended":
        return ObservedCellState.SUSPENDED
    if phase == "Pending":
        return ObservedCellState.PENDING
    serving = any(cond["type"] == "Serving" and cond["status"] == "True" for cond in cell["status"]["conditions"])
    return ObservedCellState.SERVING if serving else ObservedCellState.RUNNING_NOT_SERVING


class BaseEvent(FrozenStrictBaseModel):
    # Wall clock, so an event can be lined up against the timestamps the metric events carry.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InjectionEvent(BaseEvent):
    cell_name: str
    form_name: str
    succeeded: bool
    harmed: bool = True


class SoakActionRequest(FrozenStrictBaseModel):
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    target: dict
    form_name: str
    harms_cell: bool
    next_due_at: float | None = None


class SoakScheduleEvent(BaseEvent):
    due_of_type: dict[str, float]


class SoakActionRequestedEvent(BaseEvent):
    request: SoakActionRequest


class SoakActionResultEvent(BaseEvent):
    request_id: str
    returned: bool
    error: str | None = None


class CellInfo(FrozenStrictBaseModel):
    cell_type: str
    state: ObservedCellState
    alive: bool


class ObservationsEvent(BaseEvent):
    # One whole poll, so a cell that has vanished is as recorded as one that answered.
    cell_infos: dict[str, CellInfo]
    cells: list[dict] = Field(default_factory=list)


Event = InjectionEvent | ObservationsEvent | SoakActionRequestedEvent | SoakActionResultEvent | SoakScheduleEvent


class EventLog:
    """The fault injector's only mutable state: what happened, in order. Every question is a view of it."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._lock = threading.Lock()

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def note_injection_attempt(self, *, cell_name: str, form_name: str, succeeded: bool, harmed: bool = True) -> None:
        self._append(InjectionEvent(cell_name=cell_name, form_name=form_name, succeeded=succeeded, harmed=harmed))

    def observe(self, cells: list[dict]) -> None:
        self._append(
            ObservationsEvent(
                cells=deepcopy(cells),
                cell_infos={
                    cell["metadata"]["name"]: CellInfo(
                        cell_type=cell_type_of(cell),
                        state=compute_observed_cell_state(cell),
                        alive=cell_is_alive(cell),
                    )
                    for cell in cells
                },
            )
        )

    def note_action_requested(self, request: SoakActionRequest) -> None:
        self._append(SoakActionRequestedEvent(request=request.model_copy(deep=True)))

    def note_action_result(self, result: SoakActionResultEvent) -> None:
        self._append(result)

    def note_schedule(self, schedule: SoakScheduleEvent) -> None:
        self._append(schedule)

    def _append(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)


def cell_type_of(cell: dict) -> str:
    return cell["metadata"]["labels"]["miles.io/cell-type"]
