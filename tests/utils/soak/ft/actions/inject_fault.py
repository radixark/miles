import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import httpx
from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest
from tests.utils.soak.ft.actions.base import BaseCellFaultForm
from tests.utils.soak.ft.types import CellTarget, InjectFaultDetails, ObservedCellFault

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import FaultTarget

logger = logging.getLogger(__name__)

EFFECT_TIMEOUT_SECONDS: float = 30.0


@dataclass(frozen=True, kw_only=True)
class InjectFaultForm(BaseCellFaultForm):
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
        identity = target.fault_target
        if identity is None or identity.workers_hash != target.incarnation:
            return None
        return self._create_request(target=target, details=InjectFaultDetails(fault_target=identity))

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        assert isinstance(request.details, InjectFaultDetails), f"Request {request.request_id} names no fault target"
        fault_target = request.details.fault_target
        assert fault_target.cell_id == request.target.identity
        assert fault_target.workers_hash == request.target.incarnation
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/cells/{request.target.identity}/inject-fault",
                    json={
                        "mode": self.failure_mode.value,
                        "sub_index": fault_target.sub_index,
                        "expected_target": fault_target.model_dump(mode="json"),
                    },
                )
                if response.status_code < 500:
                    response.raise_for_status()
            except httpx.TransportError:
                logger.warning("Fault submission outcome is unknown: %s", request.request_id, exc_info=True)
            report_applied(await self._read_effect(client=client, request=request, fault_target=fault_target))

    async def _read_effect(
        self,
        *,
        client: httpx.AsyncClient,
        request: SoakActionRequest,
        fault_target: FaultTarget,
    ) -> ObservedCellFault:
        async with asyncio.timeout(EFFECT_TIMEOUT_SECONDS):
            while True:
                try:
                    response = await client.get(f"{self.base_url}/api/v1/cells/{fault_target.cell_id}")
                    if response.status_code == 404:
                        return self._effect(fault_target=fault_target, request=request, observed="missing")
                    if response.status_code < 500:
                        response.raise_for_status()
                        cell = response.json()
                        workers_hash = cell["status"]["workers_hash"]
                        if workers_hash != fault_target.workers_hash:
                            return self._effect(
                                fault_target=fault_target,
                                request=request,
                                observed="replaced",
                                observed_workers_hash=workers_hash,
                            )
                        if any(
                            condition["type"] == "Healthy" and condition["status"] == "False"
                            for condition in cell["status"]["conditions"]
                        ):
                            return self._effect(
                                fault_target=fault_target,
                                request=request,
                                observed="unhealthy",
                                observed_workers_hash=workers_hash,
                            )
                except httpx.TransportError:
                    logger.warning("Fault effect observation failed: %s", request.request_id, exc_info=True)
                await asyncio.sleep(0.2)

    def _effect(
        self,
        *,
        fault_target: FaultTarget,
        request: SoakActionRequest,
        observed: Literal["missing", "replaced", "unhealthy"],
        observed_workers_hash: str | None = None,
    ) -> ObservedCellFault:
        return ObservedCellFault(
            request_id=request.request_id,
            target=fault_target,
            mode=self.failure_mode,
            observed=observed,
            observed_workers_hash=observed_workers_hash,
        )
