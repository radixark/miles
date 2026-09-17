import asyncio
import builtins
import logging
import subprocess
import time
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any

import httpx
from tests.utils.soak.action import SoakActionError, SoakActionForm
from tests.utils.soak.config import SoakTailPolicy, SoakTimeouts
from tests.utils.soak.core import POLL_INTERVAL_SECONDS, SoakActionScheduler
from tests.utils.soak.fault_forms import CellFaultForms
from tests.utils.soak.observer import SoakObserver
from tests.utils.soak.state import (
    EventLog,
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakActionResultEvent,
    SoakAdmissionClosedEvent,
    SoakEvent,
    SoakObservation,
    target_type_of,
)
from tests.utils.soak.training_events import observe_training_events
from tests.utils.soak.views import project_actions

from miles.utils.audit_utils.event_logger.models import TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity

logger = logging.getLogger(__name__)

# A pod deletion, the slowest form, cannot be cancelled and is two kubectl calls bounded at a minute.
SHUTDOWN_TIMEOUT_SECONDS: float = 180.0


class SoakRunner:
    def __init__(
        self,
        *,
        observer: SoakObserver,
        scheduler: SoakActionScheduler,
        forms: CellFaultForms,
        event_log: EventLog,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
        timeouts: SoakTimeouts | None = None,
        training_events_dir: Path | None = None,
        tail_policy: SoakTailPolicy | None = None,
    ) -> None:
        assert poll_interval_seconds >= 0
        self.event_log = event_log
        self.cell_fault_forms = forms
        self.timeouts = timeouts if timeouts is not None else SoakTimeouts()
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
        self._timeouts = self.timeouts
        self._training_events_dir = training_events_dir
        self._tail_policy = tail_policy
        if tail_policy is not None and training_events_dir is None:
            raise ValueError("A recovery tail requires live training events")

    async def run(self, training: Coroutine[Any, Any, None], *, teardown: Callable[[], Awaitable[None]]) -> None:
        started = asyncio.Event()
        cleaning = asyncio.Event()
        owned = asyncio.create_task(
            self._run(training=training, teardown=teardown, started=started, cleaning=cleaning)
        )
        cancelled: asyncio.CancelledError | None = None
        forwarded = False
        while not owned.done():
            try:
                if cancelled is not None and not forwarded:
                    if not started.is_set():
                        await asyncio.sleep(0)
                        continue
                    if not cleaning.is_set():
                        owned.cancel()
                        forwarded = True
                await asyncio.shield(owned)
            except asyncio.CancelledError as error:
                cancelled = error
            except BaseException:
                break
        try:
            owned.result()
        except BaseException as error:
            if cancelled is not None and not forwarded and not isinstance(error, asyncio.CancelledError):
                raise builtins.BaseExceptionGroup("Soak cancellation and cleanup failed", [cancelled, error]) from None
            raise
        if cancelled is not None:
            raise cancelled

    async def _run(
        self,
        *,
        training: Coroutine[Any, Any, None],
        teardown: Callable[[], Awaitable[None]],
        started: asyncio.Event,
        cleaning: asyncio.Event,
    ) -> None:
        started.set()
        stopped = asyncio.Event()
        errors: list[BaseException] = []
        try:
            async with asyncio.TaskGroup() as tasks:
                observing = tasks.create_task(self._run_observer(stopped))
                launched = tasks.create_task(training)
                try:
                    async with asyncio.timeout(self.timeouts.run_seconds):
                        await launched
                finally:
                    self.event_log.close_admission()
                    stopped.set()
                    async with asyncio.timeout(SHUTDOWN_TIMEOUT_SECONDS):
                        await observing
        except BaseException as error:
            errors.append(error)
        finally:
            cleaning.set()
            training.close()
            try:
                self.event_log.close_admission()
            except BaseException as error:
                errors.append(error)
            for cleanup in (self.finish, teardown, self.event_log.finish):
                try:
                    await cleanup()
                except BaseException as error:
                    errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise builtins.BaseExceptionGroup("Soak session failed", errors)

    async def _run_observer(self, stop_event: asyncio.Event) -> None:
        self._event_log.note_schedule(self._scheduler.initial_schedule())
        async with asyncio.TaskGroup() as tasks:
            observing = tasks.create_task(self._observe_and_choose(stop_event))
            budget = tasks.create_task(self._watch_tail_budget(stop_event))
            await stop_event.wait()
            observing.cancel()
            budget.cancel()

    async def finish(self) -> None:
        self._finalize_cancelled_actions()
        await self._observe_and_record(timeout_seconds=self._timeouts.final_observation_seconds)
        events = self.get_events()
        for request_id, action in project_actions(events).items():
            request = action.requested.request
            form = self._forms[(target_type_of(request.target), request.form_name)]
            assert action.result is not None and action.result.returned, f"Action did not finish: {request_id}"
            assert form.is_recovered(action=action, events=events), f"Action did not recover: {request_id}"

    def get_events(self) -> list[SoakEvent]:
        return self._event_log.events

    async def _watch_tail_budget(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            closed = next((event for event in self.get_events() if isinstance(event, SoakAdmissionClosedEvent)), None)
            if closed is not None and time.monotonic() - closed.monotonic_time >= self._timeouts.tail_seconds:
                raise TimeoutError(f"Soak recovery tail exceeded {self._timeouts.tail_seconds}s")
            await asyncio.sleep(0.05)

    async def _observe_and_choose(self, stop_event: asyncio.Event) -> None:
        async with asyncio.TaskGroup() as actions:
            while not stop_event.is_set():
                await asyncio.sleep(self._poll_interval_seconds)
                # Record every poll so the post-run witnesses see the same stream the injector saw.
                await self._observe_and_record(timeout_seconds=self._timeouts.observation_seconds)
                if stop_event.is_set():
                    return
                if (request := self._scheduler.choose(events=self.get_events(), now=time.monotonic())) is None:
                    continue
                if self._event_log.note_action_requested(request):
                    actions.create_task(self._execute(request))

    async def _observe_and_record(self, *, timeout_seconds: float) -> None:
        try:
            async with asyncio.timeout(timeout_seconds):
                observation = await self._observer.observe()
                if self._training_events_dir is not None:
                    try:
                        training_events = await observe_training_events(
                            self._training_events_dir, timeout_seconds=timeout_seconds
                        )
                        observation = observation.model_copy(update={"training_events": training_events})
                    except (subprocess.SubprocessError, TimeoutError) as error:
                        logger.warning("Failed to observe training events", exc_info=True)
                        observation = observation.model_copy(
                            update={"errors": {**observation.errors, "training_events": repr(error)}}
                        )
        except TimeoutError as error:
            logger.warning("Soak observation exceeded %.1fs", timeout_seconds, exc_info=True)
            observation = SoakObservation(cells=None, errors={"observation": repr(error)})
        self._event_log.note_observation(observation)
        if self._tail_policy is not None and any(
            isinstance(event, TrainGroupStepEndEvent)
            and isinstance(event.source, TrainerControllerProcessIdentity)
            and event.source.trainer_id == self._tail_policy.trainer_id
            and event.rollout_id >= self._tail_policy.close_after_rollout_id
            for event in observation.training_events
        ):
            self._event_log.close_admission()

    async def _execute(self, request: SoakActionRequest) -> None:
        try:
            evidence = await self._forms[(target_type_of(request.target), request.form_name)].execute(request)
        except asyncio.CancelledError as error:
            self._event_log.note_action_result(
                SoakActionResultEvent(request_id=request.request_id, returned=False, error=repr(error))
            )
            raise
        except Exception as error:
            self._event_log.note_action_result(
                SoakActionResultEvent(
                    request_id=request.request_id,
                    returned=False,
                    error=repr(error),
                    evidence=error.evidence if isinstance(error, SoakActionError) else {},
                )
            )
            logger.info("Action %s failed", request.request_id, exc_info=True)
            if not isinstance(error, (httpx.HTTPError, subprocess.SubprocessError, TimeoutError)):
                raise
        else:
            if evidence is not None:
                self._event_log.note_action_applied(
                    SoakActionAppliedEvent(request_id=request.request_id, evidence=evidence)
                )
            self._event_log.note_action_result(SoakActionResultEvent(request_id=request.request_id, returned=True))

    def _finalize_cancelled_actions(self) -> None:
        for request_id, action in project_actions(self.get_events()).items():
            if action.result is None:
                self._event_log.note_action_result(
                    SoakActionResultEvent(
                        request_id=request_id,
                        returned=False,
                        error="Cancelled before execution",
                    )
                )
