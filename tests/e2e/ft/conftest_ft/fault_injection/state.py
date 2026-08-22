# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import enum
import threading
from datetime import datetime, timezone

from pydantic import Field

from miles.utils.ft_utils.api_server.models import WORKERS_HASH_LABEL
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
    workers_hash: str
    form_name: str
    succeeded: bool
    harmed: bool = True


class CellInfo(FrozenStrictBaseModel):
    cell_type: str
    workers_hash: str
    state: ObservedCellState
    alive: bool


class ObservationsEvent(BaseEvent):
    # One whole poll, so a cell that has vanished is as recorded as one that answered.
    cell_infos: dict[str, CellInfo]


Event = InjectionEvent | ObservationsEvent


class EventLog:
    """The fault injector's only mutable state: what happened, in order. Every question is a view of it."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._lock = threading.Lock()

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def note_injection_attempt(
        self,
        *,
        cell_name: str,
        workers_hash: str,
        form_name: str,
        succeeded: bool,
        harmed: bool = True,
    ) -> None:
        self._append(
            InjectionEvent(
                cell_name=cell_name,
                workers_hash=workers_hash,
                form_name=form_name,
                succeeded=succeeded,
                harmed=harmed,
            )
        )

    def observe(self, cells: list[dict]) -> None:
        self._append(
            ObservationsEvent(
                cell_infos={
                    cell["metadata"]["name"]: CellInfo(
                        cell_type=cell_type_of(cell),
                        workers_hash=cell_workers_hash(cell),
                        state=compute_observed_cell_state(cell),
                        alive=cell_is_alive(cell),
                    )
                    for cell in cells
                }
            )
        )

    def _append(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)


def cell_type_of(cell: dict) -> str:
    return cell["metadata"]["labels"]["miles.io/cell-type"]


def cell_workers_hash(cell: dict) -> str:
    return cell["metadata"]["labels"][WORKERS_HASH_LABEL]
