import asyncio
import logging
import subprocess
import time

import httpx
from tests.utils.soak.action import SoakActionForm
from tests.utils.soak.core import POLL_INTERVAL_SECONDS, SoakActionScheduler
from tests.utils.soak.fault_forms import CellFaultForms
from tests.utils.soak.observer import SoakObserver
from tests.utils.soak.state import (
    Event,
    EventLog,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    cell_type_of,
)

logger = logging.getLogger(__name__)


class SoakRunner:
    def __init__(
        self,
        *,
        observer: SoakObserver,
        scheduler: SoakActionScheduler,
        forms: CellFaultForms,
        event_log: EventLog,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        assert poll_interval_seconds >= 0
        self._observer = observer
        self._scheduler = scheduler
        self._forms: dict[tuple[str, str], SoakActionForm] = {}
        for kind, candidates in forms.items():
            for form in candidates:
                assert isinstance(form, SoakActionForm), f"Form {form.name} has no async execution"
                key = (kind, form.name)
                assert key not in self._forms, f"Duplicate action form: {key}"
                self._forms[key] = form
        self._event_log = event_log
        self._poll_interval_seconds = poll_interval_seconds

    async def run(self, stop_event: asyncio.Event) -> None:
        self._event_log.note_schedule(self._scheduler.initial_schedule())
        try:
            async with asyncio.TaskGroup() as tasks:
                observing = tasks.create_task(self._observe_and_choose(stop_event))
                await stop_event.wait()
                observing.cancel()
        finally:
            self._finalize_cancelled_actions()
            self._event_log.note_observation(await self._observer.observe())

    def get_events(self) -> list[Event]:
        return self._event_log.events

    async def _observe_and_choose(self, stop_event: asyncio.Event) -> None:
        action_task: asyncio.Task[None] | None = None
        async with asyncio.TaskGroup() as actions:
            while not stop_event.is_set():
                await asyncio.sleep(self._poll_interval_seconds)
                self._event_log.note_observation(await self._observer.observe())
                if stop_event.is_set():
                    return
                if action_task is not None and not action_task.done():
                    continue
                if (request := self._scheduler.choose(events=self.get_events(), now=time.monotonic())) is None:
                    continue
                self._event_log.note_action_requested(request)
                action_task = actions.create_task(self._execute(request))

    async def _execute(self, request: SoakActionRequest) -> None:
        try:
            await self._forms[(cell_type_of(request.target), request.form_name)].execute(request)
        except asyncio.CancelledError as error:
            self._event_log.note_action_result(
                SoakActionResultEvent(request_id=request.request_id, returned=False, error=repr(error))
            )
            raise
        except Exception as error:
            self._event_log.note_action_result(
                SoakActionResultEvent(request_id=request.request_id, returned=False, error=repr(error))
            )
            logger.info("Action %s failed", request.request_id, exc_info=True)
            if not isinstance(error, (httpx.HTTPError, subprocess.SubprocessError, TimeoutError)):
                raise
        else:
            self._event_log.note_action_result(SoakActionResultEvent(request_id=request.request_id, returned=True))

    def _finalize_cancelled_actions(self) -> None:
        events = self.get_events()
        finished = {event.request_id for event in events if isinstance(event, SoakActionResultEvent)}
        for event in events:
            if isinstance(event, SoakActionRequestedEvent) and event.request.request_id not in finished:
                self._event_log.note_action_result(
                    SoakActionResultEvent(
                        request_id=event.request.request_id,
                        returned=False,
                        error="Cancelled before execution",
                    )
                )
