import logging
import random
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest
from tests.utils.soak.ft.actions.base import assert_request_target, resolve_fault_target
from tests.utils.soak.ft.actions.inject_fault import (
    EFFECT_TIMEOUT_SECONDS,
    InjectFaultForm,
    compute_fault_hook_url,
    post_fault_hook_command,
)
from tests.utils.soak.ft.types import CellTarget, HookFaultDetails, HookFaultEvidence

from miles.utils.audit_utils.event_logger.models import FaultHookEvent
from miles.utils.test_utils.fault_injector.controller import FaultHookCommand, FaultHookOperation
from miles.utils.test_utils.fault_injector.models import (
    FaultHookName,
    FaultHookRecord,
    FaultHookRequest,
    FaultHookStatus,
)

logger = logging.getLogger(__name__)

UNFIRED_TERMINAL_STATUSES: frozenset[FaultHookStatus] = frozenset(
    {FaultHookStatus.CLEARED, FaultHookStatus.EXPIRED, FaultHookStatus.FAILED}
)


@dataclass(frozen=True, kw_only=True)
class HookFaultForm(InjectFaultForm):
    hook_name: FaultHookName
    lifetime_seconds: float
    max_delay_ms: float = 0

    @property
    def name(self) -> str:
        return f"hook:{self.hook_name}:{self.action.kind}:{self.max_delay_ms:g}ms"

    def maybe_create_request(
        self,
        *,
        target: CellTarget,
        observation: SoakObservationEvent,
        events: list[SoakEvent],
        rng: random.Random,
    ) -> SoakActionRequest | None:
        if (fault_target := resolve_fault_target(target)) is None:
            return None
        return self._create_request(
            target=target,
            details=HookFaultDetails(
                fault_target=fault_target,
                hook_name=self.hook_name,
                action=self.action,
                delay_ms=rng.uniform(0, self.max_delay_ms),
            ),
        )

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        assert isinstance(request.details, HookFaultDetails), f"Request {request.request_id} names no fault target"
        fault_target = assert_request_target(request, fault_target=request.details.fault_target)
        hook_request = FaultHookRequest(
            request_id=request.request_id,
            hook_name=self.hook_name,
            action=self.action,
            target=fault_target,
            lifetime_seconds=self.lifetime_seconds,
            delay_ms=request.details.delay_ms,
        )

        async with httpx.AsyncClient(timeout=5.0) as client, set_fault_hook(
            client=client, base_url=self.base_url, hook_request=hook_request
        ):
            effect = await self._await_effect(
                client=client,
                request=request,
                fault_target=fault_target,
                timeout_seconds=hook_request.lifetime_seconds + EFFECT_TIMEOUT_SECONDS,
            )
            report_applied(HookFaultEvidence(hook_request_id=hook_request.request_id, effect=effect))


def assert_fired_hook(event: FaultHookEvent) -> None:
    record = event.record
    assert (
        record.context is not None and record.context.debug_weight_update_id
    ), "Weight-update hook lacks its exact update"
    assert record.reached_at is not None and record.due_at is not None, "Hook lacks target-local timing"
    assert record.changed_at >= record.due_at, "Hook fired before its target-local deadline"


@asynccontextmanager
async def set_fault_hook(
    *, client: httpx.AsyncClient, base_url: str, hook_request: FaultHookRequest
) -> AsyncIterator[None]:
    set_command = FaultHookCommand(operation=FaultHookOperation.SET, request=hook_request)
    clear_command = FaultHookCommand(operation=FaultHookOperation.CLEAR, request=hook_request)

    response = await post_fault_hook_command(client, base_url=base_url, command=set_command)
    if response is not None:
        record = FaultHookRecord.model_validate(response.json())
        assert record.request == hook_request, "Set fault hook does not match the requested injection"

    try:
        yield
    finally:
        try:
            response = await client.post(
                compute_fault_hook_url(base_url=base_url, command=clear_command),
                content=clear_command.model_dump_json(),
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
