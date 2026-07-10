# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import enum
import logging
import random
import threading
import time
from collections.abc import Callable

import requests

from miles.utils.test_utils.fault_injector import FailureMode

logger = logging.getLogger(__name__)

CONTROL_SERVER_PORT: int = 18080
MEAN_INTERVAL_SECONDS: float = 60.0
# Poll cell liveness this often so the gate tracks a crash->detect->heal cycle even when it
# happens entirely between two (much sparser) injections; injections still fire on the long
# random interval above.
POLL_INTERVAL_SECONDS: float = 2.0
FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT]


def cell_is_alive(cell: dict) -> bool:
    return any(cond["type"] == "Healthy" and cond["status"] == "True" for cond in cell["status"]["conditions"])


class _CellState(enum.Enum):
    INJECTED = enum.auto()  # we crashed it; the control server may still report it Healthy
    RECOVERING = enum.auto()  # observed unhealthy; awaiting its return to Healthy


class RecoveryGate:
    def __init__(self) -> None:
        self._state_of_cell_name: dict[str, _CellState] = {}

    def note_injected(self, cell_name: str) -> None:
        self._state_of_cell_name[cell_name] = _CellState.INJECTED

    def observe(self, cells_by_name: dict[str, dict]) -> None:
        for name, state in list(self._state_of_cell_name.items()):
            cell = cells_by_name.get(name)
            if cell is None or not cell_is_alive(cell):
                self._state_of_cell_name[name] = _CellState.RECOVERING
            elif state is _CellState.RECOVERING:
                del self._state_of_cell_name[name]

    def genuinely_alive(self, cells: list[dict]) -> list[dict]:
        return [c for c in cells if cell_is_alive(c) and c["metadata"]["name"] not in self._state_of_cell_name]


def _compute_next_injection_time(rng: random.Random, mean_interval_seconds: float) -> float:
    return time.monotonic() + rng.expovariate(1.0 / mean_interval_seconds)


def run_fault_injection_loop(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds: float,
    stop_event: threading.Event,
    on_successful_injection: Callable[[], None],
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
) -> None:
    rng = random.Random(seed)
    gate = RecoveryGate()
    next_injection_time = _compute_next_injection_time(rng, mean_interval_seconds)

    while not stop_event.is_set():
        if stop_event.wait(timeout=poll_interval_seconds):
            break

        try:
            resp = requests.get(f"{base_url}/api/v1/cells", timeout=5)
            resp.raise_for_status()
            cells = resp.json()["items"]
        except Exception:
            logger.info("Failed to list cells from control server", exc_info=True)
            continue

        # Track recovery on every poll so a crash->detect->heal cycle that completes between two
        # sparse injections is seen, not missed (which would exclude the cell from the live set forever).
        gate.observe({c["metadata"]["name"]: c for c in cells})

        if time.monotonic() < next_injection_time:
            continue

        # Keep >=1 cell genuinely alive: if a prior injection has not recovered yet, wait and retry
        # on a later poll rather than killing the last live replica.
        alive = gate.genuinely_alive(cells)
        if len(alive) <= 1:
            logger.info("Deferring injection: %d genuinely-alive cell(s), need >1 to keep a live replica", len(alive))
            continue

        target = rng.choice(alive)
        cell_name = target["metadata"]["name"]
        mode = rng.choice(FAILURE_MODES)
        try:
            resp = requests.post(
                f"{base_url}/api/v1/cells/{cell_name}/inject-fault",
                json={"mode": mode.value, "sub_index": 0},
                timeout=5,
            )
            resp.raise_for_status()
            gate.note_injected(cell_name)
            on_successful_injection()
            next_injection_time = _compute_next_injection_time(rng, mean_interval_seconds)
        except Exception:
            logger.info("Failed to inject fault into %s", cell_name, exc_info=True)


class FaultInjectorHandle:
    def __init__(self, *, base_url: str, seed: int, mean_interval_seconds: float) -> None:
        self.num_successful_injections: int = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=run_fault_injection_loop,
            kwargs={
                "base_url": base_url,
                "seed": seed,
                "mean_interval_seconds": mean_interval_seconds,
                "stop_event": self._stop_event,
                "on_successful_injection": self._on_successful_injection,
            },
            daemon=True,
            name="ft-random-fault-injector",
        )

    def start(self) -> None:
        self._thread.start()

    def stop_and_join(self, *, timeout_seconds: float) -> None:
        self._stop_event.set()
        self._thread.join(timeout=timeout_seconds)

    def _on_successful_injection(self) -> None:
        self.num_successful_injections += 1


def spawn_fault_injector(*, seed: int, mean_interval_seconds: float) -> FaultInjectorHandle:
    base_url = f"http://localhost:{CONTROL_SERVER_PORT}"
    handle = FaultInjectorHandle(base_url=base_url, seed=seed, mean_interval_seconds=mean_interval_seconds)
    handle.start()
    return handle
