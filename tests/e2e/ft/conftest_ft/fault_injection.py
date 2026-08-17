# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import dataclasses
import enum
import logging
import random
import threading
import time
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Literal

import requests
from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector import FailureMode

logger = logging.getLogger(__name__)

API_SERVER_PORT: int = 18080
MEAN_INTERVAL_SECONDS: float = 60.0
# Poll cell liveness this often so the gate tracks a crash->detect->heal cycle even when it
# happens entirely between two (much sparser) injections; injections still fire on the long
# random interval above.
POLL_INTERVAL_SECONDS: float = 2.0
FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT]


def cell_is_alive(cell: dict) -> bool:
    return any(cond["type"] == "Healthy" and cond["status"] == "True" for cond in cell["status"]["conditions"])


class ObservedCellState(enum.Enum):
    SUSPENDED = "Suspended"  # torn down, holding no gpu
    PENDING = "Pending"  # allocated but gated: no engine serving yet
    RUNNING_NOT_SERVING = "RunningNotServing"  # engine is up but not registered in the router
    SERVING = "Serving"  # registered in the router, i.e. actually able to answer requests


_RELAUNCH_STATES: tuple[ObservedCellState, ...] = (ObservedCellState.SUSPENDED, ObservedCellState.PENDING)


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


class CellInfo(FrozenStrictBaseModel):
    cell_type: str
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

    def note_injected(self, cell_name: str) -> None:
        self._append(InjectionEvent(cell_name=cell_name))

    def observe(self, cells: list[dict]) -> None:
        self._append(
            ObservationsEvent(
                cell_infos={
                    cell["metadata"]["name"]: CellInfo(
                        cell_type=_cell_type_of(cell),
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


def compute_genuinely_alive(events: list[Event], cells: list[dict]) -> list[dict]:
    awaiting = compute_cells_awaiting_recovery(events)
    return [c for c in cells if cell_is_alive(c) and c["metadata"]["name"] not in awaiting]


def compute_cells_awaiting_recovery(events: list[Event]) -> set[str]:
    state_of_cell_name: dict[str, _CellState] = {}
    for event in events:
        if isinstance(event, InjectionEvent):
            state_of_cell_name[event.cell_name] = _CellState.INJECTED
            continue
        for name, state in list(state_of_cell_name.items()):
            info = event.cell_infos.get(name)
            if info is None or not info.alive:
                state_of_cell_name[name] = _CellState.RECOVERING
            elif state is _CellState.RECOVERING:
                del state_of_cell_name[name]
    return set(state_of_cell_name)


def compute_num_injections(events: list[Event], *, cell_type: str | None = None) -> int:
    return sum(
        sum(1 for one in cell_events if one.kind == "injected")
        for cell_events in _compute_matching_cell_events(events, cell_type=cell_type).values()
    )


def compute_num_completed_recoveries(events: list[Event], *, cell_type: str | None = None) -> int:
    return sum(
        _compute_recovery_tally(cell_events).num_completed
        for cell_events in _compute_matching_cell_events(events, cell_type=cell_type).values()
    )


def compute_cells_with_unfinished_recovery(events: list[Event], *, cell_type: str | None = None) -> dict[str, int]:
    return {
        name: tally.num_unfinished
        for name, cell_events in _compute_matching_cell_events(events, cell_type=cell_type).items()
        if (tally := _compute_recovery_tally(cell_events)).num_unfinished
    }


def compute_states_of_cell_name(events: list[Event]) -> dict[str, list[ObservedCellState]]:
    return {
        name: states
        for name, cell_events in _compute_cell_events(events).items()
        if (states := _compute_distinct_states(cell_events))
    }


class _CellState(enum.Enum):
    INJECTED = enum.auto()  # we crashed it; the api server may still report it Healthy
    RECOVERING = enum.auto()  # observed unhealthy; awaiting its return to Healthy


@dataclasses.dataclass(frozen=True)
class _CellEvent:
    kind: Literal["injected", "observed"]
    state: ObservedCellState | None = None


def _compute_cell_events(events: list[Event]) -> dict[str, list[_CellEvent]]:
    cell_events_of_name: dict[str, list[_CellEvent]] = {}
    for event in events:
        if isinstance(event, InjectionEvent):
            cell_events_of_name.setdefault(event.cell_name, []).append(_CellEvent(kind="injected"))
            continue
        for name, info in event.cell_infos.items():
            cell_events_of_name.setdefault(name, []).append(_CellEvent(kind="observed", state=info.state))
    return cell_events_of_name


def _compute_matching_cell_events(events: list[Event], *, cell_type: str | None) -> dict[str, list[_CellEvent]]:
    cell_events_of_name = _compute_cell_events(events)
    if cell_type is None:
        return cell_events_of_name
    cell_type_of_name = _compute_cell_type_of_name(events)
    return {
        name: cell_events
        for name, cell_events in cell_events_of_name.items()
        if cell_type_of_name.get(name) == cell_type
    }


def _compute_cell_type_of_name(events: list[Event]) -> dict[str, str]:
    cell_type_of_name: dict[str, str] = {}
    for event in events:
        if isinstance(event, ObservationsEvent):
            cell_type_of_name.update({name: info.cell_type for name, info in event.cell_infos.items()})
    return cell_type_of_name


@dataclasses.dataclass(frozen=True)
class _RecoveryTally:
    num_completed: int
    num_unfinished: int


class _RecoveryStage(enum.Enum):
    AWAITING_RELAUNCH = enum.auto()
    AWAITING_SERVING = enum.auto()


def _compute_recovery_tally(events: list[_CellEvent]) -> _RecoveryTally:
    pending: list[_RecoveryStage] = []
    num_completed = 0
    for event in events:
        if event.kind == "injected":
            pending.append(_RecoveryStage.AWAITING_RELAUNCH)
            continue
        if not pending:
            continue
        if pending[0] is _RecoveryStage.AWAITING_RELAUNCH and event.state in _RELAUNCH_STATES:
            pending[0] = _RecoveryStage.AWAITING_SERVING
        elif pending[0] is _RecoveryStage.AWAITING_SERVING and event.state is ObservedCellState.SERVING:
            pending.pop(0)
            num_completed += 1
    return _RecoveryTally(num_completed=num_completed, num_unfinished=len(pending))


def _compute_distinct_states(events: list[_CellEvent]) -> list[ObservedCellState]:
    states: list[ObservedCellState] = []
    for event in events:
        if event.kind == "observed" and event.state is not None and (not states or states[-1] != event.state):
            states.append(event.state)
    return states


def _compute_next_injection_time(rng: random.Random, mean_interval_seconds: float) -> float:
    return time.monotonic() + rng.expovariate(1.0 / mean_interval_seconds)


def run_fault_injection_loop(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds: float,
    stop_event: threading.Event,
    on_successful_injection: Callable[[], None],
    cell_type: str | None,
    event_log: EventLog,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
) -> None:
    rng = random.Random(seed)
    next_injection_time = _compute_next_injection_time(rng, mean_interval_seconds)

    while not stop_event.is_set():
        if stop_event.wait(timeout=poll_interval_seconds):
            break

        cells = list_cells(base_url=base_url, cell_type=cell_type)
        if cells is None:
            continue

        # Record every poll so a crash->detect->heal cycle that completes between two sparse
        # injections is seen, not missed (which would exclude the cell from the live set forever).
        event_log.observe(cells)

        if time.monotonic() < next_injection_time:
            continue

        # Keep >=1 cell of each kind genuinely alive: if a prior injection has not recovered yet, wait
        # and retry on a later poll rather than killing that kind's last live replica.
        alive_of_type: dict[str, list[dict]] = {}
        for cell in compute_genuinely_alive(event_log.events, cells):
            alive_of_type.setdefault(_cell_type_of(cell), []).append(cell)
        spare_types = sorted(kind for kind, kind_cells in alive_of_type.items() if len(kind_cells) > 1)
        if not spare_types:
            logger.info(
                "Deferring injection: no cell kind has a spare replica (%s)",
                {kind: len(kind_cells) for kind, kind_cells in sorted(alive_of_type.items())},
            )
            continue

        target = rng.choice(alive_of_type[rng.choice(spare_types)])
        cell_name = target["metadata"]["name"]
        mode = rng.choice(FAILURE_MODES)
        try:
            resp = requests.post(
                f"{base_url}/api/v1/cells/{cell_name}/inject-fault",
                json={"mode": mode.value, "sub_index": 0},
                timeout=5,
            )
            resp.raise_for_status()
            event_log.note_injected(cell_name)
            on_successful_injection()
            next_injection_time = _compute_next_injection_time(rng, mean_interval_seconds)
        except Exception:
            logger.info("Failed to inject fault into %s", cell_name, exc_info=True)


def list_cells(*, base_url: str, cell_type: str | None) -> list[dict] | None:
    try:
        resp = requests.get(f"{base_url}/api/v1/cells", timeout=5)
        resp.raise_for_status()
        return [c for c in resp.json()["items"] if _matches_cell_type(c, cell_type)]
    except Exception:
        logger.info("Failed to list cells from api server", exc_info=True)
        return None


def _cell_type_of(cell: dict) -> str:
    return cell["metadata"]["labels"]["miles.io/cell-type"]


def _matches_cell_type(cell: dict, cell_type: str | None) -> bool:
    return cell_type is None or _cell_type_of(cell) == cell_type


class FaultInjectorHandle:
    def __init__(self, *, base_url: str, seed: int, mean_interval_seconds: float, cell_type: str | None) -> None:
        self.num_successful_injections: int = 0
        self.event_log = EventLog()
        self._base_url = base_url
        self._cell_type = cell_type
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=run_fault_injection_loop,
            kwargs={
                "base_url": base_url,
                "seed": seed,
                "mean_interval_seconds": mean_interval_seconds,
                "stop_event": self._stop_event,
                "on_successful_injection": self._on_successful_injection,
                "cell_type": cell_type,
                "event_log": self.event_log,
            },
            daemon=True,
            name="ft-random-fault-injector",
        )

    def start(self) -> None:
        self._thread.start()

    def stop_and_join(self, *, timeout_seconds: float) -> None:
        self._stop_event.set()
        self._thread.join(timeout=timeout_seconds)
        self._observe_final_snapshot()

    def _observe_final_snapshot(self) -> None:
        cells = list_cells(base_url=self._base_url, cell_type=self._cell_type)
        if cells is None:
            return
        self.event_log.observe(cells)

    def _on_successful_injection(self) -> None:
        self.num_successful_injections += 1


def spawn_fault_injector(
    *, base_url: str, seed: int, mean_interval_seconds: float, cell_type: str | None
) -> FaultInjectorHandle:
    handle = FaultInjectorHandle(
        base_url=base_url, seed=seed, mean_interval_seconds=mean_interval_seconds, cell_type=cell_type
    )
    handle.start()
    return handle
