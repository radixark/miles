# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import threading

from tests.e2e.ft.conftest_ft.fault_injection.core import list_cells, run_fault_injection_loop
from tests.e2e.ft.conftest_ft.fault_injection.state import EventLog

API_SERVER_PORT: int = 18080
MEAN_INTERVAL_SECONDS: float = 60.0


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
