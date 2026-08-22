from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import random
from dataclasses import dataclass, field

from miles.utils.async_utils import AsyncLoopThread
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.logging_utils import configure_logger
from miles.utils.workers.registration.models import RegisteredCellInfo, RegistrationSnapshot
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo

logger = logging.getLogger(__name__)

SNAPSHOT_INTERVAL_SECONDS = 15.0
SNAPSHOT_JITTER_RATIO = 0.2
SNAPSHOT_DEBOUNCE_SECONDS = 1.0
HUB_READY_TIMEOUT_SECONDS = 3600.0
INGEST_TIMEOUT_SECONDS = 60.0


@dataclass(kw_only=True)
class RegistrationReporter:
    run_uuid: str
    reporter_id: str
    hub_endpoint: BaseWorkerHandle
    worker_provider: BaseWorkerProvider
    _trigger: _DebouncedIntervalTrigger = field(
        default_factory=lambda: _DebouncedIntervalTrigger(
            interval_seconds=SNAPSHOT_INTERVAL_SECONDS,
            jitter_ratio=SNAPSHOT_JITTER_RATIO,
            debounce_seconds=SNAPSHOT_DEBOUNCE_SECONDS,
        )
    )
    _info_of_cell_id: dict[str, CellInfo] = field(init=False, default_factory=dict)
    _sequence_number: int = field(init=False, default=0)

    async def run(self) -> None:
        await self.hub_endpoint.wait_ready(timeout=HUB_READY_TIMEOUT_SECONDS)
        stop_watch = await self.worker_provider.watch_cells(self._observe)

        try:
            logger.info(
                f"Reporter {self.reporter_id} observes {len(self._info_of_cell_id)} cells of its own deployment and "
                f"reports them every {self._trigger.interval_seconds}s"
            )
            while True:
                await self._trigger.wait()
                try:
                    await self._send_once()
                except Exception:
                    logger.warning(f"Reporting the cells of {self.reporter_id} failed", exc_info=True)
        finally:
            await stop_watch()

    async def _send_once(self) -> None:
        self._sequence_number += 1
        snapshot = self._compute_snapshot()
        await asyncio.wait_for(
            self.hub_endpoint.registration_ingest(snapshot=snapshot), timeout=INGEST_TIMEOUT_SECONDS
        )

    async def _observe(self, cell_id: str, observed: CellInfo | None) -> None:
        if observed is None:
            self._info_of_cell_id.pop(cell_id, None)
        else:
            self._info_of_cell_id[cell_id] = observed
        self._trigger.notify()

    def _compute_snapshot(self) -> RegistrationSnapshot:
        return RegistrationSnapshot(
            run_uuid=self.run_uuid,
            reporter_id=self.reporter_id,
            sequence_number=self._sequence_number,
            cells=_compute_cells(
                self._info_of_cell_id, reporter_id=self.reporter_id, worker_provider=self.worker_provider
            ),
        )


class RegistrationReporterWorker:
    def __init__(self, *, args, reporter: RegistrationReporter) -> None:
        configure_logger(args, source=SimpleProcessIdentity(component="registration_reporter"))

        self._loop_thread = AsyncLoopThread()
        self._loop_thread.submit(reporter.run()).add_done_callback(_exit_because_the_reporter_stopped)


@dataclass(kw_only=True)
class _DebouncedIntervalTrigger:
    interval_seconds: float
    jitter_ratio: float
    debounce_seconds: float
    rng: random.Random = field(default_factory=random.Random)
    _changed: asyncio.Event = field(init=False, default_factory=asyncio.Event)

    def notify(self) -> None:
        self._changed.set()

    async def wait(self) -> None:
        try:
            await asyncio.wait_for(self._changed.wait(), timeout=self._compute_next_interval_seconds())
        except (TimeoutError, asyncio.TimeoutError):
            return
        self._changed.clear()
        await asyncio.sleep(self.debounce_seconds)
        self._changed.clear()

    def _compute_next_interval_seconds(self) -> float:
        return self.interval_seconds * (1.0 + self.rng.uniform(-self.jitter_ratio, self.jitter_ratio))


def _compute_cells(
    info_of_cell_id: dict[str, CellInfo], *, reporter_id: str, worker_provider: BaseWorkerProvider
) -> list[RegisteredCellInfo]:
    observed = sorted(info_of_cell_id.items())
    worker_infos_per_cell = worker_provider.get_worker_infos(cell_ids=[cell_id for cell_id, _ in observed])
    return [
        RegisteredCellInfo(reporter_id=reporter_id, info=info, workers=worker_infos)
        for (_cell_id, info), worker_infos in zip(observed, worker_infos_per_cell, strict=True)
    ]


def _exit_because_the_reporter_stopped(future: concurrent.futures.Future[None]) -> None:
    logger.error(
        "The registration reporter of this deployment stopped, so the deployment exits with it: nothing else here "
        "registers these cells into the run, and a pod that keeps running would look healthy while the run waits "
        "for cells that are never announced",
        exc_info=future.exception(),
    )
    os._exit(1)
