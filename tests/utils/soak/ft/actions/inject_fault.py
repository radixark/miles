import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import httpx
from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest
from tests.utils.soak.ft.actions.base import BaseCellFaultForm
from tests.utils.soak.ft.cells import cell_is_alive
from tests.utils.soak.ft.types import (
    CellTarget,
    HookFaultDetails,
    InjectFaultDetails,
    ObservedCellFault,
    ObservedCellFaultKind,
)

from miles.utils.ft_utils.api_server.models import Cell, FaultInjection
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import FaultTarget

logger = logging.getLogger(__name__)

EFFECT_TIMEOUT_SECONDS: float = 30.0


@dataclass(frozen=True, kw_only=True)
class InjectFaultForm(BaseCellFaultForm):
    details_type: ClassVar[type[InjectFaultDetails | HookFaultDetails]] = InjectFaultDetails
    base_url: str
    failure_mode: FailureMode

    @property
    def name(self) -> str:
        return f"inject_fault:{self.failure_mode.value}"

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
        if (fault_target := self._resolve_fault_target(target)) is None:
            return None
        return self._create_request(target=target, details=InjectFaultDetails(fault_target=fault_target))

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        fault_target = self._assert_request_target(request)
        async with httpx.AsyncClient(timeout=5.0) as client:
            await post_fault_control(
                client,
                url=f"{self.base_url}/api/v1/cells/{request.target.identity}/inject-fault",
                body=FaultInjection(
                    mode=self.failure_mode, sub_index=fault_target.sub_index, expected_target=fault_target
                ).model_dump_json(),
                request_id=request.request_id,
            )
            report_applied(
                await self._read_effect(
                    client=client, request=request, fault_target=fault_target, timeout_seconds=EFFECT_TIMEOUT_SECONDS
                )
            )

    def _resolve_fault_target(self, target: CellTarget) -> FaultTarget | None:
        fault_target = target.fault_target
        if fault_target is None or fault_target.workers_hash != target.incarnation:
            return None
        return fault_target

    def _assert_request_target(self, request: SoakActionRequest) -> FaultTarget:
        details = request.details
        assert isinstance(details, self.details_type), f"Request {request.request_id} names no fault target"
        assert details.fault_target.cell_id == request.target.identity
        assert details.fault_target.workers_hash == request.target.incarnation
        return details.fault_target

    async def _read_effect(
        self,
        *,
        client: httpx.AsyncClient,
        request: SoakActionRequest,
        fault_target: FaultTarget,
        timeout_seconds: float,
    ) -> ObservedCellFault:
        async with asyncio.timeout(timeout_seconds):
            while (effect := await self._observe_effect_once(client=client, fault_target=fault_target)) is None:
                await asyncio.sleep(0.2)

        observed, observed_workers_hash = effect
        return ObservedCellFault(
            request_id=request.request_id,
            target=fault_target,
            mode=self.failure_mode,
            observed=observed,
            observed_workers_hash=observed_workers_hash,
        )

    async def _observe_effect_once(
        self, *, client: httpx.AsyncClient, fault_target: FaultTarget
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


async def post_fault_control(
    client: httpx.AsyncClient, *, url: str, body: str, request_id: str
) -> httpx.Response | None:
    try:
        response = await client.post(url, content=body, headers={"Content-Type": "application/json"})
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
