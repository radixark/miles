import asyncio
import logging
import random
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import ClassVar

import httpx
from tests.utils.soak.core.config import POLL_INTERVAL_SECONDS
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakActionRequestedEvent, SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest
from tests.utils.soak.core.views import alive_targets_of_kind, sut_events
from tests.utils.soak.ft.actions.base import BaseCellFaultForm
from tests.utils.soak.ft.actions.inject_fault import EFFECT_TIMEOUT_SECONDS, InjectFaultForm, post_fault_control
from tests.utils.soak.ft.types import (
    ACTOR_CELL_TYPE,
    CellTarget,
    HookFaultDetails,
    HookFaultEvidence,
    RemoteHookFaultDetails,
    RemoteHookFaultEvidence,
)

from miles.utils.audit_utils.event_logger.models import (
    Event,
    FaultHookAction,
    FaultHookEvent,
    FaultHookName,
    FaultHookRecord,
    FaultHookRequest,
    FaultHookStatus,
)
from miles.utils.ft_utils.api_server.models import FaultHookControl
from miles.utils.test_utils.fault_hooks import FaultHookCommand, FaultHookOperation
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import FaultTarget

logger = logging.getLogger(__name__)

TRAINER_EVENT_FILE_PATTERN: str = "*actor_cell*.jsonl"
HIT_GRACE_SECONDS: float = 15.0
OBSERVE_MODE: FailureMode = FailureMode.SIGKILL
TERMINAL_WITHOUT_DISPATCH: frozenset[FaultHookStatus] = frozenset(
    {FaultHookStatus.CLEARED, FaultHookStatus.EXPIRED, FaultHookStatus.FAILED}
)


@dataclass(frozen=True, kw_only=True)
class HookFaultForm(InjectFaultForm):
    details_type: ClassVar[type[HookFaultDetails]] = HookFaultDetails
    hook_name: FaultHookName
    lifetime_seconds: float
    max_delay_ms: float = 0

    @property
    def name(self) -> str:
        return f"hook:{self.hook_name}:{self.failure_mode.value}:{self.max_delay_ms:g}ms"

    def maybe_create_request(
        self,
        *,
        target: CellTarget,
        observation: SoakObservationEvent,
        events: list[SoakEvent],
        rng: random.Random,
    ) -> SoakActionRequest | None:
        if (fault_target := self._resolve_fault_target(target)) is None:
            return None
        return self._create_request(
            target=target,
            details=HookFaultDetails(
                fault_target=fault_target,
                hook_name=self.hook_name,
                mode=self.failure_mode,
                delay_ms=rng.uniform(0, self.max_delay_ms),
            ),
        )

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        fault_target = self._assert_request_target(request)
        hook_request = FaultHookRequest(
            request_id=request.request_id,
            hook_name=self.hook_name,
            mode=self.failure_mode,
            lifetime_seconds=self.lifetime_seconds,
            delay_ms=request.details.delay_ms,
        )

        async with httpx.AsyncClient(timeout=5.0) as client, _set_fault_hook(
            client=client, base_url=self.base_url, trigger=fault_target, hook_request=hook_request
        ):
            effect = await self._read_effect(
                client=client,
                request=request,
                fault_target=fault_target,
                timeout_seconds=hook_request.lifetime_seconds + EFFECT_TIMEOUT_SECONDS,
            )
            report_applied(HookFaultEvidence(hook_request_id=hook_request.request_id, effect=effect))


@dataclass(frozen=True, kw_only=True)
class RemoteHookFaultForm(BaseCellFaultForm):
    base_url: str
    hook_name: FaultHookName
    victim: BaseCellFaultForm
    event_log: EventLog
    lifetime_seconds: float
    max_delay_ms: float = 0

    @property
    def name(self) -> str:
        return f"remote_hook:{self.hook_name}:{self.victim.name}:{self.max_delay_ms:g}ms"

    @property
    def needs_fault_target(self) -> bool:
        return self.victim.needs_fault_target

    @property
    def trigger_cell_types(self) -> frozenset[str]:
        return frozenset({ACTOR_CELL_TYPE})

    @property
    def process_patterns(self) -> dict[str, str]:
        return self.victim.process_patterns

    @property
    def sut_event_file_patterns(self) -> tuple[str, ...]:
        return (TRAINER_EVENT_FILE_PATTERN,)

    @property
    def sut_event_types(self) -> tuple[type[Event], ...]:
        return (FaultHookEvent,)

    def maybe_create_request(
        self,
        *,
        target: CellTarget,
        observation: SoakObservationEvent,
        events: list[SoakEvent],
        rng: random.Random,
    ) -> SoakActionRequest | None:
        victim_request = self.victim.maybe_create_request(
            target=target, observation=observation, events=events, rng=rng
        )
        if victim_request is None:
            return None

        harmed = {
            (event.request.target.identity, event.request.target.incarnation)
            for event in events
            if isinstance(event, SoakActionRequestedEvent) and event.request.target.kind == ACTOR_CELL_TYPE
        }
        triggers = [
            trainer.fault_target
            for trainer in alive_targets_of_kind(observation, ACTOR_CELL_TYPE)
            if trainer.identity != target.identity
            and trainer.fault_target is not None
            and trainer.fault_target.workers_hash == trainer.incarnation
            and (trainer.identity, trainer.incarnation) not in harmed
        ]
        if not triggers:
            return None

        return self._create_request(
            target=target,
            details=RemoteHookFaultDetails(
                trigger=rng.choice(triggers),
                hook_name=self.hook_name,
                delay_ms=rng.uniform(0, self.max_delay_ms),
                victim_form=self.victim.name,
                victim=victim_request.details,
            ),
        )

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        assert isinstance(request.details, RemoteHookFaultDetails), f"Request {request.request_id} names no trigger"
        trigger = request.details.trigger
        assert trigger.cell_id != request.target.identity, "Remote hook must target another cell"
        hook_request = FaultHookRequest(
            request_id=f"{request.request_id}:trigger",
            hook_name=self.hook_name,
            mode=OBSERVE_MODE,
            action=FaultHookAction.OBSERVE,
            lifetime_seconds=self.lifetime_seconds,
            delay_ms=request.details.delay_ms,
        )

        async with httpx.AsyncClient(timeout=5.0) as client, _set_fault_hook(
            client=client, base_url=self.base_url, trigger=trigger, hook_request=hook_request
        ):
            hit = await self._wait_for_hit(hook_request)
            victim_evidence = await self._execute_victim(
                request.model_copy(update={"form_name": self.victim.name, "details": request.details.victim})
            )
            report_applied(RemoteHookFaultEvidence(hit=hit, victim_evidence=victim_evidence))

    async def _wait_for_hit(self, hook_request: FaultHookRequest) -> FaultHookEvent:
        async with asyncio.timeout(hook_request.lifetime_seconds + HIT_GRACE_SECONDS):
            while True:
                for event in sut_events(self.event_log.events):
                    if (
                        not isinstance(event, FaultHookEvent)
                        or event.record.request.request_id != hook_request.request_id
                    ):
                        continue
                    if event.record.status == FaultHookStatus.FIRED:
                        assert_fired_hook(event)
                        return event
                    if event.record.status in TERMINAL_WITHOUT_DISPATCH:
                        raise RuntimeError(f"Remote fault trigger cannot fire: {event.record.status}")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _execute_victim(self, request: SoakActionRequest) -> SoakActionEvidence:
        evidence: list[SoakActionEvidence] = []
        await self.victim.execute(request, report_applied=evidence.append)
        assert len(evidence) == 1, "Remote hook victim supplied no effect evidence"
        return evidence[0]


def assert_fired_hook(event: FaultHookEvent) -> None:
    record = event.record
    assert (
        record.context is not None and record.context.debug_weight_update_id
    ), "Weight-update hook lacks its exact update"
    assert record.reached_at is not None and record.due_at is not None, "Hook lacks target-local timing"
    assert record.changed_at >= record.due_at, "Hook fired before its target-local deadline"


@asynccontextmanager
async def _set_fault_hook(
    *, client: httpx.AsyncClient, base_url: str, trigger: FaultTarget, hook_request: FaultHookRequest
) -> AsyncIterator[None]:
    endpoint = f"{base_url}/api/v1/cells/{trigger.cell_id}/fault-hook"

    response = await post_fault_control(
        client,
        url=endpoint,
        body=_control(trigger=trigger, operation=FaultHookOperation.SET, hook_request=hook_request),
        request_id=hook_request.request_id,
    )
    if response is not None:
        record = FaultHookRecord.model_validate(response.json())
        assert record.request == hook_request, "Set fault hook does not match the requested injection"

    try:
        yield
    finally:
        try:
            response = await client.post(
                endpoint,
                content=_control(trigger=trigger, operation=FaultHookOperation.CLEAR, hook_request=hook_request),
                headers={"Content-Type": "application/json"},
            )
            if response.status_code not in {httpx.codes.CONFLICT, httpx.codes.PRECONDITION_FAILED}:
                response.raise_for_status()
        except httpx.HTTPError:
            logger.warning(
                "Fault hook clearing is unconfirmed; its lifetime still bounds dispatch: %s",
                hook_request.request_id,
                exc_info=True,
            )


def _control(*, trigger: FaultTarget, operation: FaultHookOperation, hook_request: FaultHookRequest) -> str:
    return FaultHookControl(
        target=trigger, command=FaultHookCommand(operation=operation, request=hook_request)
    ).model_dump_json()
