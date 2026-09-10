import asyncio
import logging
import random

import httpx
from tests.utils.soak.action import SoakActionError, SoakActionForm
from tests.utils.soak.batch import validate_fault_batch
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
        victim_form: SoakActionForm | None = None,
        all_targets: bool = False,
        random_delay: bool = False,
    ) -> None:
        super().__init__(base_url=base_url, failure_mode=failure_mode)
        self._victim_form = victim_form
        if all_targets and victim_form is None:
            raise ValueError("All-target hooks require a remote victim form")
        self.all_targets = all_targets
        if random_delay and failure_mode == FailureMode.THREAD_DEADLOCK and victim_form is None:
            raise ValueError("Training-thread deadlock requires an immediate hook")
        self.random_delay = random_delay
        self._template = FaultHookRequest(
            request_id="template",
            instance_id="template",
            hook=hook,
            mode=failure_mode.value,
            delay_ms=delay_ms,
            lifetime_seconds=lifetime_seconds,
            action="inject" if victim_form is None else "observe",
        )

    @property
    def victim_form(self) -> SoakActionForm | None:
        return self._victim_form

    @property
    def name(self) -> str:
        delay = "random" if self.random_delay else f"{self._template.delay_ms:g}ms"
        if self._victim_form is not None:
            suffix = ":all" if self.all_targets else ""
            return f"remote_hook:{self._template.hook}:{self._victim_form.name}:{delay}{suffix}"
        return f"hook:{self._template.hook}:{self._failure_mode.value}:{delay}"

    def sample_delay(self, rng: random.Random) -> float | None:
        return rng.uniform(0, self._template.delay_ms) if self.random_delay else None

    async def execute(self, request: SoakActionRequest) -> dict:
        assert request.form_name == self.name
        if self.random_delay:
            assert request.hook_delay_ms is not None and 0 <= request.hook_delay_ms <= self._template.delay_ms
        else:
            assert request.hook_delay_ms is None
        assert isinstance(request.target, dict)
        if self.all_targets:
            validate_fault_batch(request)
            assert request.additional_requests, "All-target hooks require multiple victims"
        else:
            assert not request.additional_requests, "This hook does not accept a fault batch"
        trigger = request.fault_target if self._victim_form is None else request.hook_trigger
        assert trigger is not None, "Hook requires an observed trigger process"
        if self._victim_form is None:
            assert trigger.cell_id == request.target["metadata"]["name"]
            assert trigger.workers_hash == request.target["status"]["workers_hash"]
        else:
            assert trigger.cell_id != request.target["metadata"]["name"], "Remote hook must target another cell"
        endpoint = f"{self._base_url}/api/v1/cells/{trigger.cell_id}/fault-hook"
        target = trigger.model_dump(mode="json")

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                endpoint, json={"target": target, "command": FaultHookCommand(operation="inspect").model_dump()}
            )
            response.raise_for_status()
            instance_id = response.json()
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError("Fault hook inspection did not return a process incarnation")
            hook_request = self._template.model_copy(
                update={
                    "request_id": request.request_id if self._victim_form is None else f"{request.request_id}:trigger",
                    "instance_id": instance_id,
                    "delay_ms": request.hook_delay_ms if self.random_delay else self._template.delay_ms,
                }
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

                if self._victim_form is not None:
                    hit = await self._wait_for_hit(
                        client=client, endpoint=endpoint, target=target, request=hook_request
                    )
                    if self.all_targets:
                        expected = {
                            child.target["metadata"]["name"]: child.target["status"]["workers_hash"]
                            for child in [request, *request.additional_requests]
                        }
                        assert hit.target_incarnations == expected, "Sender assignment changed before batch injection"
                    receipt = await self._execute_victims(request)
                    return {
                        **receipt,
                        "hook_request": hook_request.model_dump(mode="json"),
                        "hook_hit": hit.model_dump(mode="json"),
                        "hook_trigger": target,
                        "victim_form": self._victim_form.name,
                    }
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

    async def _execute_victims(self, request: SoakActionRequest) -> dict:
        assert self._victim_form is not None
        assert request.hook_trigger is not None
        requests = [request, *request.additional_requests]
        assert all(child.form_name == self._victim_form.name for child in request.additional_requests)
        assert all(child.target["metadata"]["name"] != request.hook_trigger.cell_id for child in requests)
        outcomes = await asyncio.gather(
            *[
                self._victim_form.execute(
                    child.model_copy(update={"form_name": self._victim_form.name, "additional_requests": []})
                )
                for child in requests
            ],
            return_exceptions=True,
        )
        if request.additional_requests and any(
            outcome is None or isinstance(outcome, BaseException) for outcome in outcomes
        ):
            failures = [
                child.request_id
                for child, outcome in zip(requests, outcomes, strict=True)
                if outcome is None or isinstance(outcome, BaseException)
            ]
            raise SoakActionError(
                f"Remote hook victims failed: {failures}",
                evidence={
                    "victim_outcomes": {
                        child.request_id: (
                            {"error": repr(outcome)}
                            if isinstance(outcome, BaseException)
                            else {"error": "No effect receipt"} if outcome is None else {"receipt": outcome}
                        )
                        for child, outcome in zip(requests, outcomes, strict=True)
                    }
                },
            )
        for child, outcome in zip(requests, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome is None:
                raise RuntimeError(f"Remote hook victim supplied no effect evidence: {child.request_id}")
        receipt = outcomes[0]
        if request.additional_requests:
            receipt = {
                **receipt,
                "batch_receipts": {
                    child.request_id: outcome
                    for child, outcome in zip(request.additional_requests, outcomes[1:], strict=True)
                },
            }
        return receipt

    async def _wait_for_hit(
        self, *, client: httpx.AsyncClient, endpoint: str, target: dict, request: FaultHookRequest
    ) -> FaultHookRecord:
        async with asyncio.timeout(request.lifetime_seconds + 15.0):
            while True:
                try:
                    response = await client.post(
                        endpoint,
                        json={
                            "target": target,
                            "command": FaultHookCommand(operation="read", request=request).model_dump(mode="json"),
                        },
                    )
                    if response.status_code != 404 and response.status_code < 500:
                        response.raise_for_status()
                        record = FaultHookRecord.model_validate(response.json())
                        if record.request.model_copy(update={"receipt_url": None}) != request:
                            raise ValueError("Remote fault trigger belongs to another hook request")
                        if record.status == "fired":
                            if (
                                not record.update_id
                                or record.weight_version is None
                                or record.reached_at is None
                                or record.due_at is None
                            ):
                                raise ValueError("Remote fault trigger lacks weight-update timing evidence")
                            if not record.due_at <= record.changed_at < record.armed_at + request.lifetime_seconds:
                                raise ValueError("Remote fault trigger fired outside its valid timing window")
                            return record
                        if record.status in {"cancelled", "expired", "failed"}:
                            raise RuntimeError(f"Remote fault trigger cannot fire: {record.status}")
                except httpx.TransportError:
                    logger.warning("Remote fault trigger read failed: %s", request.request_id, exc_info=True)
                await asyncio.sleep(0.05)
