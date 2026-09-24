import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest
from tests.utils.soak.ft.actions.base import BaseCellFaultForm
from tests.utils.soak.ft.cells import cell_is_alive
from tests.utils.soak.ft.types import CellTarget, InjectFaultDetails, ObservedCellFault, ObservedCellFaultKind

from miles.utils.ft_utils.api_server.models import Cell
from miles.utils.test_utils.fault_injector.actions.union import FaultAction
from miles.utils.test_utils.fault_injector.controller import FaultHookCommand, FaultHookOperation
from miles.utils.test_utils.fault_injector.models import FaultHookName, FaultHookRequest, ObservedFaultHookTarget

logger = logging.getLogger(__name__)

EFFECT_TIMEOUT_SECONDS: float = 30.0


@dataclass(frozen=True, kw_only=True)
class InjectFaultForm(BaseCellFaultForm):
    base_url: str
    action: FaultAction
    hook_name: FaultHookName | None = None
    lifetime_seconds: float | None = None
    max_delay_ms: float = 0

    @property
    def name(self) -> str:
        name = f"inject_fault:{self.action.kind}"
        if self.hook_name is not None:
            name += f":{self.hook_name}:{self.max_delay_ms:g}ms"
        return name

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
                hook_target=identity,
                hook_name=self.hook_name,
                delay_ms=rng.uniform(0, self.max_delay_ms),
            ),
        )

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        assert isinstance(request.details, InjectFaultDetails), f"Request {request.request_id} names no fault target"
        details = request.details
        fault_target = details.fault_target
        assert fault_target.cell_id == request.target.identity
        assert fault_target.workers_hash == request.target.incarnation
        command = FaultHookCommand(
            operation=FaultHookOperation.SET,
            request=FaultHookRequest(
                request_id=request.request_id,
                hook_name=details.hook_name,
                action=self.action,
                target=details.hook_target,
                lifetime_seconds=self.lifetime_seconds,
                delay_ms=details.delay_ms,
            ),
        )
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/cells/{details.hook_target.cell_id}/fault-hook",
                    content=command.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )
                if response.is_server_error:
                    logger.warning(
                        "Fault submission outcome is unknown for %s: HTTP %s %s",
                        request.request_id,
                        response.status_code,
                        response.text,
                    )
                else:
                    response.raise_for_status()
            except httpx.TransportError:
                logger.warning("Fault submission outcome is unknown: %s", request.request_id, exc_info=True)
            report_applied(await self._read_effect(client=client, request=request, fault_target=fault_target))

    async def _read_effect(
        self,
        *,
        client: httpx.AsyncClient,
        request: SoakActionRequest,
        fault_target: ObservedFaultHookTarget,
    ) -> ObservedCellFault:
        async with asyncio.timeout(EFFECT_TIMEOUT_SECONDS + (self.lifetime_seconds or 0)):
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
