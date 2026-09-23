import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from tests.utils.soak.core.archive import archive_evidence
from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import (
    SoakActionRequestedEvent,
    SoakAdmissionClosedEvent,
    SoakEvent,
    SoakObservationEvent,
)
from tests.utils.soak.core.scheduler import SoakActionScheduler
from tests.utils.soak.core.sut_events import SutEventFeed
from tests.utils.soak.core.types import SoakForms, SoakObserver
from tests.utils.soak.core.views import admission_closed, trainer_step_ends

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SoakRunner:
    observer: SoakObserver
    scheduler: SoakActionScheduler
    forms: SoakForms
    event_log: EventLog
    config: SoakRunnerConfig
    sut_events: SutEventFeed | None = None

    def __post_init__(self) -> None:
        if self.config.tail is not None and self.sut_events is None:
            raise ValueError("A recovery tail requires live training events")

    async def run(self, sut_run: Awaitable[object], *, teardown: Callable[[], Awaitable[None]]) -> None:
        try:
            await self._run(sut_run=sut_run)
        finally:
            self._close_admission()
            await self._finish()
            await teardown()
            await archive_evidence(self.event_log)

    async def _finish(self) -> None:
        await self._observe_and_record(timeout_seconds=self.config.timeouts.final_observation_seconds)

    async def _run(self, *, sut_run: Awaitable[object]) -> None:
        async with asyncio.TaskGroup() as tasks:
            observing = tasks.create_task(self._observe_and_choose())
            async with asyncio.timeout(self.config.timeouts.run_seconds):
                await sut_run
            observing.cancel()

    async def _observe_and_choose(self) -> None:
        while True:
            await asyncio.sleep(self.config.poll_interval_seconds)
            await self._observe_and_record(timeout_seconds=self.config.timeouts.observation_seconds)

            events = self.event_log.events
            _assert_within_tail_budget(events, tail_seconds=self.config.timeouts.tail_seconds)

            if (request := self.scheduler.choose(events=events, now=time.monotonic())) is None:
                continue

            self.event_log.append(SoakActionRequestedEvent(request=request))

    async def _observe_and_record(self, *, timeout_seconds: float) -> None:
        try:
            async with asyncio.timeout(timeout_seconds):
                observation = await self.observer.observe()
                if self.sut_events is not None:
                    observation = await self.sut_events.attach(observation)
        except TimeoutError as error:
            logger.warning("Soak observation exceeded %.1fs", timeout_seconds, exc_info=True)
            observation = SoakObservationEvent(targets=None, errors={"observation": repr(error)})
        self.event_log.append(observation)
        if (tail := self.config.tail) is not None and any(
            step.rollout_id >= tail.close_after_rollout_id for step in trainer_step_ends([observation])
        ):
            self._close_admission()

    def _close_admission(self) -> None:
        if admission_closed(self.event_log.events) is None:
            self.event_log.append(SoakAdmissionClosedEvent())


def _assert_within_tail_budget(events: list[SoakEvent], *, tail_seconds: float) -> None:
    if (closed := admission_closed(events)) is None:
        return
    if datetime.now(timezone.utc) - closed.timestamp >= timedelta(seconds=tail_seconds):
        raise TimeoutError(f"Soak recovery tail exceeded {tail_seconds}s")
