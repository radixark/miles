import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest
from tests.utils.soak.ft.actions.base import BaseCellFaultForm, assert_request_target, resolve_fault_target
from tests.utils.soak.ft.cells import cell_is_alive
from tests.utils.soak.ft.types import CellTarget, InjectFaultDetails, ObservedCellFault, ObservedCellFaultKind

from miles.utils.ft_utils.api_server.models import Cell
from miles.utils.test_utils.fault_injector.actions.union import FaultAction
from miles.utils.test_utils.fault_injector.controller import FaultHookCommand, FaultHookOperation
from miles.utils.test_utils.fault_injector.models import FaultHookRequest, ObservedFaultHookTarget

logger = logging.getLogger(__name__)

EFFECT_TIMEOUT_SECONDS: float = 30.0


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
        if (fault_target := resolve_fault_target(target)) is None:
            return None
        return self._create_request(target=target, details=InjectFaultDetails(fault_target=fault_target))

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        assert isinstance(request.details, InjectFaultDetails), f"Request {request.request_id} names no fault target"
        fault_target = assert_request_target(request, fault_target=request.details.fault_target)
        command = FaultHookCommand(
            operation=FaultHookOperation.SET,
            request=FaultHookRequest(request_id=request.request_id, action=self.action, target=fault_target),
        )
        async with httpx.AsyncClient(timeout=5.0) as client:
            await post_fault_hook_command(client, base_url=self.base_url, command=command)
            report_applied(
                await self._await_effect(
                    client=client, request=request, fault_target=fault_target, timeout_seconds=EFFECT_TIMEOUT_SECONDS
                )
            )

    async def _await_effect(
        self,
        *,
        client: httpx.AsyncClient,
        request: SoakActionRequest,
        fault_target: ObservedFaultHookTarget,
        timeout_seconds: float,
    ) -> ObservedCellFault:
        async with asyncio.timeout(timeout_seconds):
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


async def post_fault_hook_command(
    client: httpx.AsyncClient, *, base_url: str, command: FaultHookCommand
) -> httpx.Response | None:
    request_id = command.request.request_id
    try:
        response = await client.post(
            compute_fault_hook_url(base_url=base_url, command=command),
            content=command.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
    except httpx.TransportError:
        logger.warning("Fault control outcome is unknown: %s", request_id, exc_info=True)
        return None
    if response.is_server_error:
        logger.warning(
            "Fault control outcome is unknown for %s: HTTP %s %s", request_id, response.status_code, response.text
        )
        return None
    response.raise_for_status()
    return response


def compute_fault_hook_url(*, base_url: str, command: FaultHookCommand) -> str:
    target = command.request.target
    assert isinstance(target, ObservedFaultHookTarget), f"Fault hook {command.request.request_id} names no cell"
    return f"{base_url}/api/v1/cells/{target.cell_id}/fault-hook"
