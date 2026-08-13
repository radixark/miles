# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging
import random
import threading
import time
import requests

from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import ROLLOUT_CELL_TYPE, BaseFaultForm, CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.state import (
    Event,
    EventLog,
    ObservedCellState,
    cell_type_of,
    compute_observed_cell_state,
)
from tests.e2e.ft.conftest_ft.fault_injection.views import compute_genuinely_alive, compute_successful_form_names

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

        # Record every poll so a crash->detect->heal cycle that completes between two sparse
        # injections is seen, not missed (which would exclude the cell from the live set forever).
        event_log.observe(cells)

        if stop_event.is_set():
            break

        now: float = time.monotonic()
        due_types = sorted(kind for kind, due_at in next_injection_time_of_cell_type.items() if now >= due_at)
        if not due_types:
            continue

        # Keep >=1 cell of each kind genuinely alive: if a prior injection has not recovered yet, wait
        # and retry on a later poll rather than killing that kind's last live replica.
        live_replicas_of_type: dict[str, list[dict]] = {}
        victims_of_type: dict[str, list[dict]] = {}
        for cell in compute_genuinely_alive(event_log.events, cells):
            kind = cell_type_of(cell)
            victims_of_type.setdefault(kind, []).append(cell)
            if _cell_can_serve(cell):
                live_replicas_of_type.setdefault(kind, []).append(cell)

        logger.info(
            "Live replicas %s, injectable victims %s",
            {
                kind: sorted(c["metadata"]["name"] for c in kind_cells)
                for kind, kind_cells in sorted(live_replicas_of_type.items())
            },
            {
                kind: sorted(c["metadata"]["name"] for c in kind_cells)
                for kind, kind_cells in sorted(victims_of_type.items())
            },
        )

        spare_types = [kind for kind in due_types if len(live_replicas_of_type.get(kind, [])) > 1]
        if not spare_types:
            logger.info(
                "Deferring injection: no due cell kind has a spare working replica (due %s, alive %s)",
                due_types,
                {kind: len(kind_cells) for kind, kind_cells in sorted(live_replicas_of_type.items())},
            )
            continue

        cell_type = rng.choice(spare_types)
        target = rng.choice(victims_of_type[cell_type])
        cell_name = target["metadata"]["name"]
        form = _draw_form(cell_fault_forms[cell_type], events=event_log.events, cell_type=cell_type, rng=rng)
        try:
            form.inject(target, rng)
        except Exception:
            event_log.note_injection_attempt(cell_name=cell_name, form_name=form.name, succeeded=False)
            logger.info("Failed to inject fault %s into %s", form.name, cell_name, exc_info=True)
            continue

        event_log.note_injection_attempt(cell_name=cell_name, form_name=form.name, succeeded=True)
        next_injection_time_of_cell_type[cell_type] = _compute_next_injection_time(
            rng, mean_interval_seconds_of_cell_type[cell_type]
        )
        logger.info("Injected fault %s into %s", form.name, cell_name)


def _cell_can_serve(cell: dict) -> bool:
    if cell_type_of(cell) != ROLLOUT_CELL_TYPE:
        return True
    return compute_observed_cell_state(cell) is ObservedCellState.SERVING


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
