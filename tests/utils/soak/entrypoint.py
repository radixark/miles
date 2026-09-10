# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import asyncio
import random
import threading
from collections.abc import Callable

from tests.utils.soak.core import POLL_INTERVAL_SECONDS, SoakActionScheduler, list_cells, run_fault_injection_loop
from tests.utils.soak.fault_forms import CellFaultForms
from tests.utils.soak.observer import SoakObserver
from tests.utils.soak.runner import SoakRunner
from tests.utils.soak.state import EventLog

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.test_utils.polling_worker import PollingWorker
from miles.utils.workers.types import ClusterBackend, DeployComponent

API_SERVER_PORT: int = 18080
# A pod deletion, the slowest form, cannot be cancelled and is two kubectl calls bounded at a minute.
STOP_AND_JOIN_TIMEOUT_SECONDS: float = 180.0


class FaultInjectorHandle:
    def __init__(
        self,
        *,
        base_url: str,
        seed: int,
        mean_interval_seconds_of_cell_type: dict[str, float],
        cell_fault_forms: CellFaultForms,
        get_virtual_cells: Callable[[], list[dict]] | None = None,
        injection_enabled: Callable[[], bool] | None = None,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
        namespace: str | None = None,
        release: str | None = None,
        event_log: EventLog | None = None,
        observer: SoakObserver | None = None,
    ) -> None:
        self.event_log = event_log if event_log is not None else EventLog()
        self.cell_fault_forms = cell_fault_forms
        self._base_url = base_url
        self._cell_types: set[str] = set(mean_interval_seconds_of_cell_type)
        self._get_virtual_cells: Callable[[], list[dict]] | None = get_virtual_cells
        self._runner = (
            SoakRunner(
                observer=(
                    observer
                    if observer is not None
                    else SoakObserver(
                        base_url=base_url, cell_types=self._cell_types, namespace=namespace, release=release
                    )
                ),
                scheduler=SoakActionScheduler(
                    rng=random.Random(seed),
                    mean_intervals=mean_interval_seconds_of_cell_type,
                    forms=cell_fault_forms,
                    injection_enabled=injection_enabled,
                ),
                forms={kind: cell_fault_forms[kind] for kind in self._cell_types},
                event_log=self.event_log,
                poll_interval_seconds=poll_interval_seconds,
            )
            if get_virtual_cells is None
            else None
        )

        def inject_until_stopped(stop_event: threading.Event) -> None:
            if self._runner is not None:
                asyncio.run(self._run_async(stop_event))
                return
            run_fault_injection_loop(
                base_url=base_url,
                seed=seed,
                mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
                stop_event=stop_event,
                event_log=self.event_log,
                cell_fault_forms=cell_fault_forms,
                get_virtual_cells=get_virtual_cells,
                injection_enabled=injection_enabled,
                poll_interval_seconds=poll_interval_seconds,
            )

        self._worker = PollingWorker(name="ft-random-fault-injector", run=inject_until_stopped)

    def start(self) -> None:
        self._worker.start()

    def raise_if_failed(self) -> None:
        self._worker.join(timeout_seconds=0)

    def stop_and_join(self) -> None:
        self._worker.stop_and_join(timeout_seconds=STOP_AND_JOIN_TIMEOUT_SECONDS)
        self._worker.assert_not_running(
            message=(
                f"The fault injector was still mid-injection {STOP_AND_JOIN_TIMEOUT_SECONDS}s after being asked to "
                f"stop: it may still crash a cell nothing will heal, and reading its log would race it"
            )
        )
        if self._runner is None:
            self._observe_final_snapshot()

    async def _run_async(self, stop_event: threading.Event) -> None:
        assert self._runner is not None
        stopped = asyncio.Event()
        async with asyncio.TaskGroup() as tasks:
            forwarding = tasks.create_task(_forward_stop(source=stop_event, target=stopped))
            try:
                await self._runner.run(stopped)
            finally:
                forwarding.cancel()

    def _observe_final_snapshot(self) -> None:
        cells = list_cells(base_url=self._base_url, cell_types=self._cell_types)
        if cells is None:
            return
        if self._get_virtual_cells is not None:
            cells.extend(self._get_virtual_cells())
        self.event_log.observe(cells)


def spawn_fault_injector(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds_of_cell_type: dict[str, float],
    cell_fault_forms: CellFaultForms,
    get_virtual_cells: Callable[[], list[dict]] | None = None,
    injection_enabled: Callable[[], bool] | None = None,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    config: ExecuteTrainConfig | None = None,
    event_log: EventLog | None = None,
    observer: SoakObserver | None = None,
) -> FaultInjectorHandle:
    use_kubernetes = config is not None and config.cluster_backend is ClusterBackend.KUBERNETES
    handle = FaultInjectorHandle(
        event_log=event_log,
        observer=observer,
        base_url=base_url,
        seed=seed,
        mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
        cell_fault_forms=cell_fault_forms,
        get_virtual_cells=get_virtual_cells,
        injection_enabled=injection_enabled,
        poll_interval_seconds=poll_interval_seconds,
        namespace=config.namespace if use_kubernetes else None,
        release=(
            ReleaseName(
                run_id=config.run_id, deploy_component=DeployComponent.ALL, deploy_instance_id=None
            ).serialize()
            if use_kubernetes
            else None
        ),
    )
    handle.start()
    return handle


async def _forward_stop(*, source: threading.Event, target: asyncio.Event) -> None:
    while not source.is_set():
        await asyncio.sleep(0.05)
    target.set()
