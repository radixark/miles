import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent, SoakRunContextEvent
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest
from tests.utils.soak.ft.actions.base import BaseCellFaultForm
from tests.utils.soak.ft.cells import cell_is_alive
from tests.utils.soak.ft.types import CellTarget, InjectFaultDetails, ObservedCellFault, ObservedCellFaultKind

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import Event, FaultHookEvent
from miles.utils.ft_utils.api_server.models import Cell
from miles.utils.test_utils.fault_injector.actions.union import FaultAction
from miles.utils.test_utils.fault_injector.controller import FaultHookCommand, FaultHookOperation
from miles.utils.test_utils.fault_injector.models import FaultHookRequest, FaultHookStatus, ObservedFaultHookTarget

logger = logging.getLogger(__name__)

EFFECT_TIMEOUT_SECONDS: float = 120.0


@dataclass(frozen=True, kw_only=True)
class InjectFaultForm(BaseCellFaultForm):
    base_url: str
    action: FaultAction

    @property
    def name(self) -> str:
        return f"inject_fault:{self.action.kind}"

    @property
    def needs_fault_target(self) -> bool:
        return True

    def maybe_create_request(
        self,
        *,
        target: CellTarget,
        observation: SoakObservationEvent,
        events: list[SoakEvent],
        rng: random.Random,
    ) -> SoakActionRequest | None:
        identity = target.fault_target
        if identity is None or identity.workers_hash != target.incarnation:
            return None
        return self._create_request(
            target=target,
            details=InjectFaultDetails(
                fault_target=identity,
                training_events_dir=next(
                    (
                        event.sources.get("training_events")
                        for event in reversed(events)
                        if isinstance(event, SoakRunContextEvent)
                    ),
                    None,
                ),
            ),
        )

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        assert isinstance(request.details, InjectFaultDetails), f"Request {request.request_id} names no fault target"
        fault_target = request.details.fault_target
        assert fault_target.cell_id == request.target.identity
        assert fault_target.workers_hash == request.target.incarnation
        command = FaultHookCommand(
            operation=FaultHookOperation.SET,
            request=FaultHookRequest(request_id=request.request_id, action=self.action, target=fault_target),
        )
        async with (
            httpx.AsyncClient(timeout=5.0) as client,
            asyncio.timeout(EFFECT_TIMEOUT_SECONDS),
        ):
            reader = None
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/cells/{request.target.identity}/fault-hook",
                    content=command.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                if isinstance(error, httpx.HTTPStatusError) and not error.response.is_server_error:
                    raise
                if request.details.training_events_dir is None:
                    raise
                logger.warning(
                    "Fault response was lost; checking dispatch evidence for %s", request.request_id, exc_info=True
                )
                reader = request.details.training_events_dir
                while not _dispatch_confirmed(await asyncio.to_thread(read_events, reader), request=command.request):
                    await asyncio.sleep(0.2)

            effect = await self._read_effect(client=client, request=request, fault_target=fault_target)
            if reader is not None:
                assert _dispatch_confirmed(await asyncio.to_thread(read_events, reader), request=command.request)
            report_applied(effect)

    async def _read_effect(
        self,
        *,
        client: httpx.AsyncClient,
        request: SoakActionRequest,
        fault_target: ObservedFaultHookTarget,
    ) -> ObservedCellFault:
        while (effect := await self._observe_effect_once(client=client, fault_target=fault_target)) is None:
            await asyncio.sleep(0.2)

        observed, observed_workers_hash = effect
        return ObservedCellFault(
            request_id=request.request_id,
            target=fault_target,
            action=self.action,
            observed=observed,
            observed_workers_hash=observed_workers_hash,
        )

    async def _observe_effect_once(
        self, *, client: httpx.AsyncClient, fault_target: ObservedFaultHookTarget
    ) -> tuple[ObservedCellFaultKind, str | None] | None:
        try:
            response = await client.get(f"{self.base_url}/api/v1/cells/{fault_target.cell_id}")
        except httpx.TransportError:
            logger.warning("Fault effect observation failed for %s", fault_target.cell_id, exc_info=True)
            return None

        if response.status_code == 404:
            return ObservedCellFaultKind.MISSING, None
        if response.is_server_error:
            logger.warning(
                "Fault effect is unreadable for %s: HTTP %s %s",
                fault_target.cell_id,
                response.status_code,
                response.text,
            )
            return None
        response.raise_for_status()

        cell = Cell.model_validate(response.json())
        if cell.status.workers_hash != fault_target.workers_hash:
            return ObservedCellFaultKind.REPLACED, cell.status.workers_hash
        if not cell_is_alive(cell):
            return ObservedCellFaultKind.UNHEALTHY, cell.status.workers_hash
        return None


def _dispatch_confirmed(events: list[Event], *, request: FaultHookRequest) -> bool:
    records = [
        event.record
        for event in events
        if isinstance(event, FaultHookEvent) and event.record.request.request_id == request.request_id
    ]
    assert all(record.request == request for record in records), "Dispatch evidence names another fault request"
    assert not any(record.status == FaultHookStatus.FAILED for record in records), "Fault dispatch failed"
    fired = [record for record in records if record.status == FaultHookStatus.FIRED]
    assert len(fired) <= 1, "Fault request was dispatched more than once"
    return len(fired) == 1
