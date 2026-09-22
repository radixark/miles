import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from tests.utils.soak.core import archive
from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import (
    SoakActionRequestedEvent,
    SoakAdmissionClosedEvent,
    SoakEvent,
    SoakObservationEvent,
)
from tests.utils.soak.core.scheduler import POLL_INTERVAL_SECONDS, SoakActionScheduler
from tests.utils.soak.core.sut_events import SutEventFeed
from tests.utils.soak.core.types import SoakActionRequest, SoakForms, SoakObserver
from tests.utils.soak.core.views import admission_closed, trainer_step_ends

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT_SECONDS: float = 180.0


@dataclass(kw_only=True)
class SoakRunner:
    observer: SoakObserver
    scheduler: SoakActionScheduler
    forms: SoakForms
    event_log: EventLog
    config: SoakRunnerConfig
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS
    sut_events: SutEventFeed | None = None

    def __post_init__(self) -> None:
        assert self.poll_interval_seconds >= 0
        if self.config.tail is not None and self.sut_events is None:
            raise ValueError("A recovery tail requires live training events")

    async def run(self, training: Coroutine[Any, Any, Any], *, teardown: Callable[[], Awaitable[None]]) -> None:
        try:
            await self._run(training=training)
        finally:
            self._close_admission()
            await self.finish()
            await teardown()
            await archive.finish(self.event_log)

    async def finish(self) -> None:
        await self._observe_and_record(timeout_seconds=self.config.timeouts.final_observation_seconds)

    async def _run(self, *, training: Coroutine[Any, Any, Any]) -> None:
        stopped = asyncio.Event()
        async with asyncio.TaskGroup() as tasks:
            observing = tasks.create_task(self._observe_and_choose(stopped))
            launched = tasks.create_task(training)
            try:
                async with asyncio.timeout(self.config.timeouts.run_seconds):
                    await asyncio.shield(launched)
            except TimeoutError as error:
                raise TimeoutError(f"Soak training exceeded {self.config.timeouts.run_seconds}s") from error
            finally:
                stopped.set()
                async with asyncio.timeout(SHUTDOWN_TIMEOUT_SECONDS):
                    await observing

    async def _observe_and_choose(self, stop_event: asyncio.Event) -> None:
        self.event_log.append(self.scheduler.initial_schedule())
        async with asyncio.TaskGroup() as actions:
            while not stop_event.is_set():
                await asyncio.sleep(self.poll_interval_seconds)
                await self._observe_and_record(timeout_seconds=self.config.timeouts.observation_seconds)
                events = self.event_log.events
                self._assert_within_tail_budget(events)
                if stop_event.is_set():
                    return
                if (request := self.scheduler.choose(events=events, now=time.monotonic())) is None:
                    continue
                self.event_log.append(SoakActionRequestedEvent(request=request))
                actions.create_task(self._execute(request))

    def _assert_within_tail_budget(self, events: list[SoakEvent]) -> None:
        if (closed := admission_closed(events)) is None:
            return
        if datetime.now(timezone.utc) - closed.timestamp >= timedelta(seconds=self.config.timeouts.tail_seconds):
            raise TimeoutError(f"Soak recovery tail exceeded {self.config.timeouts.tail_seconds}s")

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

    async def _execute(self, request: SoakActionRequest) -> None:
        raise NotImplementedError
