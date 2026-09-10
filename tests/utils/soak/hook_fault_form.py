import logging
import random

import httpx
from tests.utils.soak.fault_forms import InjectFaultForm
from tests.utils.soak.state import SoakActionRequest

from miles.utils.test_utils.fault_hooks import FaultHookCommand, FaultHookName, FaultHookRecord, FaultHookRequest
from miles.utils.test_utils.fault_injector import FailureMode

logger = logging.getLogger(__name__)


class HookFaultForm(InjectFaultForm):
    def __init__(
        self,
        *,
        base_url: str,
        failure_mode: FailureMode,
        hook: FaultHookName,
        delay_ms: float = 0.0,
        lifetime_seconds: float = 60.0,
    ) -> None:
        super().__init__(base_url=base_url, failure_mode=failure_mode)
        self._template = FaultHookRequest(
            request_id="template",
            instance_id="template",
            hook=hook,
            mode=failure_mode.value,
            delay_ms=delay_ms,
            lifetime_seconds=lifetime_seconds,
        )

    @property
    def name(self) -> str:
        return f"hook:{self._template.hook}:{self._failure_mode.value}:{self._template.delay_ms:g}ms"

    async def execute(self, request: SoakActionRequest) -> dict:
        assert request.form_name == self.name
        assert isinstance(request.target, dict)
        assert request.fault_target is not None
        assert request.fault_target.cell_id == request.target["metadata"]["name"]
        assert request.fault_target.workers_hash == request.target["status"]["workers_hash"]
        endpoint = f"{self._base_url}/api/v1/cells/{request.fault_target.cell_id}/fault-hook"
        target = request.fault_target.model_dump(mode="json")

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                endpoint, json={"target": target, "command": FaultHookCommand(operation="inspect").model_dump()}
            )
            response.raise_for_status()
            instance_id = response.json()
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError("Fault hook inspection did not return a process incarnation")
            hook_request = self._template.model_copy(
                update={"request_id": request.request_id, "instance_id": instance_id}
            )

            try:
                try:
                    response = await client.post(
                        endpoint,
                        json={
                            "target": target,
                            "command": FaultHookCommand(operation="arm", request=hook_request).model_dump(mode="json"),
                        },
                    )
                    if response.status_code < 500:
                        response.raise_for_status()
                        record = FaultHookRecord.model_validate(response.json())
                        if record.request.model_copy(update={"receipt_url": None}) != hook_request:
                            raise ValueError("Armed fault hook does not match the requested injection")
                        if record.status in {"cancelled", "expired", "failed"}:
                            raise RuntimeError(f"Fault hook cannot execute: {record.status}")
                except httpx.TransportError:
                    logger.warning("Fault hook arm outcome is unknown: %s", request.request_id, exc_info=True)

                receipt = await self._read_receipt(
                    client=client, request=request, timeout_seconds=hook_request.lifetime_seconds + 30.0
                )
                return {**receipt, "hook_request": hook_request.model_dump(mode="json")}
            finally:
                try:
                    response = await client.post(
                        endpoint,
                        json={
                            "target": target,
                            "command": FaultHookCommand(operation="cancel", request=hook_request).model_dump(
                                mode="json"
                            ),
                        },
                    )
                    response.raise_for_status()
                except httpx.HTTPError:
                    logger.warning(
                        "Fault hook cancellation is unconfirmed; target lease still bounds dispatch: %s",
                        request.request_id,
                        exc_info=True,
                    )

    def inject(self, cell: dict, rng: random.Random) -> None:
        raise NotImplementedError("Hook injection requires an observed asynchronous request")
