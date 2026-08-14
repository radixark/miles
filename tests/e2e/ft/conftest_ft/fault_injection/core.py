# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging
import random
import threading
import time
from collections.abc import Callable

import requests

from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.state import EventLog, cell_type_of
from tests.e2e.ft.conftest_ft.fault_injection.views import compute_genuinely_alive

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS: float = 2.0


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
    cell_fault_forms: CellFaultForms,
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
            alive_of_type.setdefault(cell_type_of(cell), []).append(cell)
        spare_types = sorted(kind for kind, kind_cells in alive_of_type.items() if len(kind_cells) > 1)
        if not spare_types:
            logger.info(
                "Deferring injection: no cell kind has a spare replica (%s)",
                {kind: len(kind_cells) for kind, kind_cells in sorted(alive_of_type.items())},
            )
            continue

        target = rng.choice(alive_of_type[rng.choice(spare_types)])
        cell_name = target["metadata"]["name"]
        form = rng.choice(cell_fault_forms[cell_type_of(target)])
        try:
            form.inject(target, rng)
            event_log.note_injected(cell_name)
            on_successful_injection()
            next_injection_time = _compute_next_injection_time(rng, mean_interval_seconds)
            logger.info("Injected fault %s into %s", form.name, cell_name)
        except Exception:
            logger.info("Failed to inject fault %s into %s", form.name, cell_name, exc_info=True)


def list_cells(*, base_url: str, cell_type: str | None) -> list[dict] | None:
    try:
        resp = requests.get(f"{base_url}/api/v1/cells", timeout=5)
        resp.raise_for_status()
        return [c for c in resp.json()["items"] if _matches_cell_type(c, cell_type)]
    except Exception:
        logger.info("Failed to list cells from api server", exc_info=True)
        return None


def _matches_cell_type(cell: dict, cell_type: str | None) -> bool:
    return cell_type is None or cell_type_of(cell) == cell_type
