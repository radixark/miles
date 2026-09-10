# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging
import random
import threading
import time
from collections.abc import Callable

import requests
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import BaseFaultForm, CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.state import Event, EventLog, cell_is_alive, cell_type_of
from tests.e2e.ft.conftest_ft.fault_injection.views import (
    compute_successful_form_names,
    compute_unrecovered_injected_cells,
)

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
    next_injection_time_of_cell_type: dict[str, float] = {
        cell_type: _compute_next_injection_time(rng, mean_interval_seconds)
        for cell_type, mean_interval_seconds in sorted(mean_interval_seconds_of_cell_type.items())
    }

    while not stop_event.is_set():
        if stop_event.wait(timeout=poll_interval_seconds):
            break

        cells = list_cells(base_url=base_url, cell_types=set(mean_interval_seconds_of_cell_type))
        if cells is None:
            continue
        if get_virtual_cells is not None:
            cells.extend(get_virtual_cells())

        # Record every poll so the post-run witnesses see the same stream the injector saw.
        event_log.observe(cells)

        if stop_event.is_set():
            break

        cells_of_type: dict[str, list[dict]] = {cell_type: [] for cell_type in next_injection_time_of_cell_type}
        for cell in cells:
            cells_of_type[cell_type_of(cell)].append(cell)
        now: float = time.monotonic()
        due_types = sorted(kind for kind, due_at in next_injection_time_of_cell_type.items() if now >= due_at)
        if not due_types:
            continue

        unrecovered = compute_unrecovered_injected_cells(event_log.events, cell_type="actor")
        if "actor" in cells_of_type:
            cells_of_type["actor"] = [
                cell
                for cell in cells_of_type["actor"]
                if cell["metadata"]["name"] not in unrecovered and cell_is_alive(cell)
            ]
        ready_types = [kind for kind in due_types if len(cells_of_type[kind]) > 1]
        if not ready_types:
            continue

        cell_type = rng.choice(ready_types)
        target = rng.choice(cells_of_type[cell_type])
        cell_name = target["metadata"]["name"]
        form = _draw_form(cell_fault_forms[cell_type], events=event_log.events, cell_type=cell_type, rng=rng)
        if injection_enabled is not None and not injection_enabled():
            continue
        next_injection_time_of_cell_type[cell_type] = _compute_next_injection_time(
            rng, mean_interval_seconds_of_cell_type[cell_type]
        )
        try:
            form.inject(target, rng)
        except Exception:
            event_log.note_injection_attempt(
                cell_name=cell_name, form_name=form.name, succeeded=False, harmed=form.harms_cell
            )
            logger.info("Failed to inject fault %s into %s", form.name, cell_name, exc_info=True)
            continue

        event_log.note_injection_attempt(
            cell_name=cell_name, form_name=form.name, succeeded=True, harmed=form.harms_cell
        )
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
