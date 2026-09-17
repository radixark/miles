import asyncio
import logging
import random

import httpx
from tests.utils.soak.action import SoakActionForm
from tests.utils.soak.fault_forms import InjectFaultForm
from tests.utils.soak.state import (
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakDeploymentTarget,
    SoakEvent,
    SoakObservation,
    cell_is_alive,
    cell_type_of,
)

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
        random_delay: bool = False,
    ) -> None:
        super().__init__(base_url=base_url, failure_mode=failure_mode)
        self._victim_form = victim_form
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
            return f"remote_hook:{self._template.hook}:{self._victim_form.name}:{delay}"
        return f"hook:{self._template.hook}:{self._failure_mode.value}:{delay}"

    def sample_delay(self, rng: random.Random) -> float | None:
        return rng.uniform(0, self._template.delay_ms) if self.random_delay else None

    def fault_target_types(self, kind: str) -> set[str]:
        return {kind} if self._victim_form is None else {"actor"} | self._victim_form.fault_target_types(kind)

    @property
    def process_patterns(self) -> dict[str, str]:
        return {} if self._victim_form is None else self._victim_form.process_patterns

    def prepare_request(
        self,
        *,
        target: dict | SoakDeploymentTarget,
        observation: SoakObservation,
        events: list[SoakEvent],
        rng: random.Random,
    ) -> SoakActionRequest | None:
        assert isinstance(target, dict)
        prepare = super().prepare_request if self._victim_form is None else self._victim_form.prepare_request
        request = prepare(target=target, observation=observation, events=events, rng=rng)
        if request is None:
            return None
        trigger = None
        if self._victim_form is not None:
            reserved = {
                (event.request.target["metadata"]["name"], event.request.target["status"].get("workers_hash"))
                for event in events
                if isinstance(event, SoakActionRequestedEvent)
                and event.request.harms_cell
                and isinstance(event.request.target, dict)
            }
            triggers = [
                identity
                for cell in observation.cells or []
                if cell_type_of(cell) == "actor"
                and cell_is_alive(cell)
                and cell["metadata"]["name"] != target["metadata"]["name"]
                if (identity := observation.fault_targets.get(cell["metadata"]["name"])) is not None
                and identity.workers_hash == cell["status"].get("workers_hash")
                and (identity.cell_id, identity.workers_hash) not in reserved
            ]
            if not triggers:
                return None
            trigger = rng.choice(triggers)
        return request.model_copy(
            update={
                "form_name": self.name,
                "hook_trigger": trigger,
                "hook_delay_ms": self.sample_delay(rng),
            }
        )

    async def execute(self, request: SoakActionRequest) -> dict:
        assert request.form_name == self.name
        if self.random_delay:
            assert request.hook_delay_ms is not None and 0 <= request.hook_delay_ms <= self._template.delay_ms
        else:
            assert request.hook_delay_ms is None
        assert isinstance(request.target, dict)
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
                        if record.request != hook_request:
                            raise ValueError("Armed fault hook does not match the requested injection")
                        if record.status in {"cancelled", "expired", "failed"}:
                            raise RuntimeError(f"Fault hook cannot execute: {record.status}")
                except httpx.TransportError:
                    logger.warning("Fault hook arm outcome is unknown: %s", request.request_id, exc_info=True)

                if self._victim_form is not None:
                    hit = await self._wait_for_hit(
                        client=client, endpoint=endpoint, target=target, request=hook_request
                    )
                    receipt = await self._victim_form.execute(
                        request.model_copy(update={"form_name": self._victim_form.name})
                    )
                    assert receipt is not None, "Remote hook victim supplied no effect evidence"
                    return {
                        **receipt,
                        "hook_request": hook_request.model_dump(mode="json"),
                        "hook_hit": hit.model_dump(mode="json"),
                        "hook_trigger": target,
                        "victim_form": self._victim_form.name,
                    }
                receipt = await self._read_effect(
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
                        if record.request != request:
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
