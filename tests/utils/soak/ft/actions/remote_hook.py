import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from tests.utils.soak.core.config import POLL_INTERVAL_SECONDS
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakActionRequestedEvent, SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest
from tests.utils.soak.core.views import alive_targets_of_kind, sut_events
from tests.utils.soak.ft.actions.base import BaseCellFaultForm, resolve_fault_target
from tests.utils.soak.ft.actions.hook import UNFIRED_TERMINAL_STATUSES, assert_fired_hook, set_fault_hook
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, CellTarget, RemoteHookFaultDetails, RemoteHookFaultEvidence

from miles.utils.audit_utils.event_logger.models import Event, FaultHookEvent
from miles.utils.test_utils.fault_injector.actions.process import ObserveAction
from miles.utils.test_utils.fault_injector.models import FaultHookName, FaultHookRequest, FaultHookStatus

logger = logging.getLogger(__name__)

TRAINER_EVENT_FILE_PATTERN: str = "*actor_cell*.jsonl"
HIT_FEED_LAG_SECONDS: float = 15.0


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
            fault_target
            for trainer in alive_targets_of_kind(observation, ACTOR_CELL_TYPE)
            if trainer.identity != target.identity
            and (trainer.identity, trainer.incarnation) not in harmed
            and (fault_target := resolve_fault_target(trainer)) is not None
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
            action=ObserveAction(),
            target=trigger,
            lifetime_seconds=self.lifetime_seconds,
            delay_ms=request.details.delay_ms,
        )

        async with httpx.AsyncClient(timeout=5.0) as client, set_fault_hook(
            client=client, base_url=self.base_url, hook_request=hook_request
        ):
            hit = await self._wait_for_hit(hook_request)
            victim_evidence = await self._execute_victim(
                request.model_copy(update={"form_name": self.victim.name, "details": request.details.victim})
            )
            report_applied(RemoteHookFaultEvidence(hit=hit, victim_evidence=victim_evidence))

    async def _wait_for_hit(self, hook_request: FaultHookRequest) -> FaultHookEvent:
        async with asyncio.timeout(hook_request.lifetime_seconds + HIT_FEED_LAG_SECONDS):
            while True:
                hits = [
                    event
                    for event in sut_events(self.event_log.events)
                    if isinstance(event, FaultHookEvent) and event.record.request.request_id == hook_request.request_id
                ]
                if fired := [event for event in hits if event.record.status == FaultHookStatus.FIRED]:
                    assert_fired_hook(fired[0])
                    return fired[0]
                if unfired := [event for event in hits if event.record.status in UNFIRED_TERMINAL_STATUSES]:
                    raise RuntimeError(f"Remote fault trigger cannot fire: {unfired[0].record.status}")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _execute_victim(self, request: SoakActionRequest) -> SoakActionEvidence:
        evidence: list[SoakActionEvidence] = []
        await self.victim.execute(request, report_applied=evidence.append)
        assert len(evidence) == 1, "Remote hook victim supplied no effect evidence"
        return evidence[0]
