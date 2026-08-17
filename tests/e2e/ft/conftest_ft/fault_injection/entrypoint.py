# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import threading

from tests.e2e.ft.conftest_ft.fault_injection.core import list_cells, run_fault_injection_loop
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.state import EventLog
from tests.e2e.ft.conftest_ft.fault_injection.views import compute_num_injections

API_SERVER_PORT: int = 18080
MEAN_INTERVAL_SECONDS: float = 60.0
# A pod deletion, the slowest form, cannot be cancelled and is two kubectl calls bounded at a minute.
STOP_AND_JOIN_TIMEOUT_SECONDS: float = 180.0


class FaultInjectorHandle:
    def __init__(
        self,
        *,
        base_url: str,
        seed: int,
        mean_interval_seconds: float,
        cell_type: str | None,
        cell_fault_forms: CellFaultForms,
    ) -> None:
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
                "cell_type": cell_type,
                "event_log": self.event_log,
                "cell_fault_forms": cell_fault_forms,
            },
            daemon=True,
            name="ft-random-fault-injector",
        )

    @property
    def num_successful_injections(self) -> int:
        return compute_num_injections(self.event_log.events)

    def start(self) -> None:
        self._thread.start()

    def stop_and_join(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=STOP_AND_JOIN_TIMEOUT_SECONDS)
        self._observe_final_snapshot()

    def _observe_final_snapshot(self) -> None:
        cells = list_cells(base_url=self._base_url, cell_type=self._cell_type)
        if cells is None:
            return
        self.event_log.observe(cells)


def spawn_fault_injector(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds: float,
    cell_type: str | None,
    cell_fault_forms: CellFaultForms,
) -> FaultInjectorHandle:
    handle = FaultInjectorHandle(
        base_url=base_url,
        seed=seed,
        mean_interval_seconds=mean_interval_seconds,
        cell_type=cell_type,
        cell_fault_forms=cell_fault_forms,
    )
    handle.start()
    return handle
