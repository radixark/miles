# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging
import random
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import requests
from tests.utils.soak.fault_forms import BaseFaultForm, CellFaultForms
from tests.utils.soak.state import (
    Event,
    EventLog,
    ObservationsEvent,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakDeploymentTarget,
    SoakObservation,
    SoakScheduleEvent,
    cell_type_of,
    target_type_of,
)
from tests.utils.soak.views import compute_successful_form_names

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS: float = 2.0


def _compute_next_injection_time(rng: random.Random, mean_interval_seconds: float) -> float:
    return time.monotonic() + rng.expovariate(1.0 / mean_interval_seconds)


def run_fault_injection_loop(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds_of_cell_type: dict[str, float],
    stop_event: threading.Event,
    event_log: EventLog,
    cell_fault_forms: CellFaultForms,
    get_virtual_cells: Callable[[], list[dict]] | None = None,
    injection_enabled: Callable[[], bool] | None = None,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
) -> None:
    rng = random.Random(seed)
    observer = _SynchronousObserver(
        base_url=base_url, cell_types=set(mean_interval_seconds_of_cell_type), get_virtual_cells=get_virtual_cells
    )
    scheduler = SoakActionScheduler(
        rng=rng,
        mean_intervals=mean_interval_seconds_of_cell_type,
        forms=cell_fault_forms,
        injection_enabled=injection_enabled,
    )
    event_log.note_schedule(scheduler.initial_schedule())

    while not stop_event.is_set():
        if stop_event.wait(timeout=poll_interval_seconds):
            break

        cells = observer.observe()
        if cells is None:
            continue

        # Record every poll so the post-run witnesses see the same stream the injector saw.
        event_log.observe(cells)

        if stop_event.is_set():
            break

        if (action := scheduler.choose(events=event_log.events, now=time.monotonic())) is not None:
            _execute_action(action=action, forms=cell_fault_forms, rng=rng, event_log=event_log)


@dataclass(frozen=True)
class _SynchronousObserver:
    base_url: str
    cell_types: set[str]
    get_virtual_cells: Callable[[], list[dict]] | None = None

    def observe(self) -> list[dict] | None:
        cells = list_cells(base_url=self.base_url, cell_types=self.cell_types)
        if cells is None:
            return None
        if self.get_virtual_cells is not None:
            cells.extend(self.get_virtual_cells())
        return cells


class SoakActionScheduler:
    def __init__(
        self,
        *,
        rng: random.Random,
        mean_intervals: dict[str, float],
        forms: CellFaultForms,
        injection_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._rng = rng
        self._mean_intervals = mean_intervals
        self._forms = forms
        self._injection_enabled = injection_enabled

    def initial_schedule(self) -> SoakScheduleEvent:
        return SoakScheduleEvent(
            due_of_type={
                cell_type: _compute_next_injection_time(self._rng, mean_interval_seconds)
                for cell_type, mean_interval_seconds in sorted(self._mean_intervals.items())
            }
        )

    def choose(self, *, events: list[Event], now: float) -> SoakActionRequest | None:
        due_of_type: dict[str, float] = {}
        observation = None
        for event in events:
            if isinstance(event, SoakScheduleEvent):
                due_of_type.update(event.due_of_type)
            elif isinstance(event, SoakActionRequestedEvent) and event.request.next_due_at is not None:
                due_of_type[target_type_of(event.request.target)] = event.request.next_due_at
            elif isinstance(event, (ObservationsEvent, SoakObservation)):
                observation = event
        if observation is None:
            return None
        cells_of_type: dict[str, list[dict | SoakDeploymentTarget]] = {
            cell_type: [] for cell_type in self._mean_intervals
        }
        for cell in observation.cells or []:
            if cell_type_of(cell) in cells_of_type:
                cells_of_type[cell_type_of(cell)].append(cell)
        if isinstance(observation, SoakObservation) and "deployment" in cells_of_type:
            cells_of_type["deployment"].extend(observation.deployments)
        due_types = sorted(kind for kind, due_at in due_of_type.items() if now >= due_at)
        if not due_types:
            return None

        ready_types = [kind for kind in due_types if len(cells_of_type[kind]) >= (1 if kind == "deployment" else 2)]
        if not ready_types:
            return None

        cell_type = self._rng.choice(ready_types)
        target = self._rng.choice(cells_of_type[cell_type])
        form = _draw_form(self._forms[cell_type], events=events, cell_type=cell_type, rng=self._rng)
        if self._injection_enabled is not None and not self._injection_enabled():
            return None
        candidates = None
        if isinstance(observation, SoakObservation) and form.name in {"delete_pod", "exec_sigkill"}:
            assert isinstance(target, dict), "Pod faults require a cell target"
            candidates = observation.pods_of_cell.get(target["metadata"]["name"], [])
            if not candidates:
                return None
        next_due_at = _compute_next_injection_time(self._rng, self._mean_intervals[cell_type])
        pod = self._rng.choice(candidates) if candidates is not None else None
        return SoakActionRequest(
            target=deepcopy(target),
            form_name=form.name,
            harms_cell=form.harms_cell,
            next_due_at=next_due_at,
            pod=pod,
        )


def _execute_action(
    *, action: SoakActionRequest, forms: CellFaultForms, rng: random.Random, event_log: EventLog
) -> None:
    assert isinstance(action.target, dict), "The synchronous bridge only supports cell targets"
    matching = [form for form in forms[cell_type_of(action.target)] if form.name == action.form_name]
    assert len(matching) == 1, f"Expected one form named {action.form_name}, found {len(matching)}"
    form = matching[0]
    cell_name = action.target["metadata"]["name"]
    request = action
    event_log.note_action_requested(request)
    try:
        form.inject(action.target, rng)
    except Exception as error:
        event_log.note_action_result(
            SoakActionResultEvent(request_id=request.request_id, returned=False, error=repr(error))
        )
        logger.info("Failed to inject fault %s into %s", form.name, cell_name, exc_info=True)
        return

    event_log.note_action_result(SoakActionResultEvent(request_id=request.request_id, returned=True))
    logger.info("Injected fault %s into %s", form.name, cell_name)


def _draw_form(
    forms: list[BaseFaultForm], *, events: list[Event], cell_type: str, rng: random.Random
) -> BaseFaultForm:
    worked = compute_successful_form_names(events, cell_type=cell_type)
    unproven = [form for form in forms if form.name not in worked]
    return rng.choice(unproven or forms)


def list_cells(*, base_url: str, cell_types: set[str]) -> list[dict] | None:
    try:
        resp = requests.get(f"{base_url}/api/v1/cells", timeout=5)
        resp.raise_for_status()
        return [c for c in resp.json()["items"] if cell_type_of(c) in cell_types]
    except Exception:
        logger.info("Failed to list cells from api server", exc_info=True)
        return None
