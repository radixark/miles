from typing import Literal
from uuid import uuid4

import httpx
from pydantic import Field

from miles.utils.test_utils.fault_injector.actions.base import BaseFaultAction, FaultHookContext, FaultHookResources
from miles.utils.test_utils.fault_injector.actions.process import ProcessFaultAction

API_SERVER_TIMEOUT_SECONDS: float = 30.0


class ApiServerFaultAction(BaseFaultAction):
    kind: Literal["api_server_fault"] = "api_server_fault"
    base_url: str
    cell_id: str
    rank: int = Field(ge=0)
    workers_hash: str | None = Field(default=None, min_length=1)
    inner: ProcessFaultAction

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        from miles.utils.test_utils.fault_injector.controller import FaultHookCommand, FaultHookOperation
        from miles.utils.test_utils.fault_injector.models import FaultHookRequest, ObservedFaultHookTarget
        from miles.utils.workers.cell_operations.base import StaleFaultTargetError

        async with httpx.AsyncClient(timeout=API_SERVER_TIMEOUT_SECONDS) as client:
            observed = await client.get(
                f"{self.base_url}/api/v1/cells/{self.cell_id}/fault-target", params={"rank": self.rank}
            )
            observed.raise_for_status()
            target = ObservedFaultHookTarget.model_validate(observed.json())
            if self.workers_hash not in (None, target.workers_hash):
                raise StaleFaultTargetError(
                    f"Cell {self.cell_id} was replaced after the fault named incarnation {self.workers_hash}"
                )
            command = FaultHookCommand(
                operation=FaultHookOperation.SET,
                request=FaultHookRequest(request_id=f"{self.kind}_{uuid4().hex}", action=self.inner, target=target),
            )
            response = await client.post(
                f"{self.base_url}/api/v1/cells/{self.cell_id}/fault-hook",
                content=command.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            if response.is_client_error:
                response.raise_for_status()
