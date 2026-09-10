import asyncio
import json

import httpx
import pytest
from tests.fast.utils.soak.utils import typed_cell
from tests.utils.soak.action import SoakActionError
from tests.utils.soak.fault_forms import InjectFaultForm
from tests.utils.soak.hook_fault_form import HookFaultForm
from tests.utils.soak.state import SoakActionRequest

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import FaultTarget


class TestHookFaultForm:
    async def test_cancelling_batch_waits_for_every_victim_to_unwind(
        self, batch_request: SoakActionRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancelling a batch collects every pending victim operation without fabricating receipts."""
        form = HookFaultForm(
            base_url="http://control",
            failure_mode=FailureMode.SIGKILL,
            hook="trainer_before_weight_send",
            victim_form=InjectFaultForm(base_url="http://control", failure_mode=FailureMode.SIGKILL),
            all_targets=True,
        )
        started: set[str] = set()
        cancelled: set[str] = set()
        ready = asyncio.Event()

        async def respond(http_request: httpx.Request) -> httpx.Response:
            assert http_request.method == "POST"
            request_id = json.loads(http_request.content)["request_id"]
            started.add(request_id)
            if len(started) == 2:
                ready.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                await asyncio.sleep(0)
                cancelled.add(request_id)
                raise
            raise AssertionError("Unreachable response")

        client_type = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs)
        )
        task = asyncio.create_task(form._execute_victims(batch_request))
        try:
            await asyncio.wait_for(ready.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            assert cancelled == started == {"victim-0", "victim-1"}
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    @pytest.mark.parametrize("failed", [False, True])
    async def test_batch_dispatches_every_victim_before_waiting_for_receipts(
        self, batch_request: SoakActionRequest, monkeypatch: pytest.MonkeyPatch, failed: bool
    ) -> None:
        """Batch victims start concurrently and one failed receipt prevents an overall applied result."""
        form = HookFaultForm(
            base_url="http://control",
            failure_mode=FailureMode.SIGKILL,
            hook="trainer_before_weight_send",
            victim_form=InjectFaultForm(base_url="http://control", failure_mode=FailureMode.SIGKILL),
            all_targets=True,
        )
        requests = {child.request_id: child for child in [batch_request, *batch_request.additional_requests]}
        started: set[str] = set()
        read: set[str] = set()
        all_started = asyncio.Event()

        async def respond(http_request: httpx.Request) -> httpx.Response:
            if http_request.method == "POST":
                body = json.loads(http_request.content)
                started.add(body["request_id"])
                if started == set(requests):
                    all_started.set()
                await asyncio.wait_for(all_started.wait(), timeout=1)
                return httpx.Response(status_code=200)
            request_id = http_request.url.path.rsplit("/", 1)[1]
            child = requests[request_id]
            read.add(request_id)
            return httpx.Response(
                status_code=200,
                json={
                    "request_id": "wrong-receipt" if failed and request_id == "victim-1" else request_id,
                    "target": child.fault_target.model_dump(mode="json"),
                    "mode": "sigkill",
                    "exited_pids": [42],
                },
            )

        client_type = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs)
        )
        if failed:
            with pytest.raises(SoakActionError, match="victim-1") as caught:
                await form._execute_victims(batch_request)
            outcomes = caught.value.evidence["victim_outcomes"]
            assert outcomes["victim-0"]["receipt"]["request_id"] == "victim-0"
            assert "does not match" in outcomes["victim-1"]["error"]
        else:
            receipt = await form._execute_victims(batch_request)
            assert receipt["request_id"] == "victim-0"
            assert receipt["batch_receipts"]["victim-1"]["target"] == requests["victim-1"].fault_target.model_dump(
                mode="json"
            )
        assert started == read == set(requests)

    @pytest.mark.parametrize(
        "outcome",
        [
            "hit",
            "expired",
            "wrong_instance",
            "stale_trigger",
            "wrong_victim",
            "missing_update",
            "changed_assignment",
        ],
    )
    async def test_remote_injection_requires_matching_trigger_then_matching_victim_receipt(
        self, monkeypatch: pytest.MonkeyPatch, outcome: str
    ) -> None:
        """A trainer hit authorizes only the recorded inference victim and cannot replace its effect receipt."""
        victim_form = InjectFaultForm(base_url="http://control", failure_mode=FailureMode.SIGKILL)
        form = HookFaultForm(
            base_url="http://control",
            failure_mode=FailureMode.SIGKILL,
            hook="trainer_before_weight_send",
            victim_form=victim_form,
            all_targets=outcome == "changed_assignment",
        )
        trigger = FaultTarget(cell_id="actor-0", sub_index=0, workers_hash="trainer-generation")
        victim = FaultTarget(cell_id="rollout-1", sub_index=0, workers_hash="generation-0")
        request = SoakActionRequest(
            form_name=form.name,
            target=typed_cell("rollout-1", "rollout"),
            harms_cell=True,
            fault_target=victim,
            hook_trigger=trigger,
        )
        operations: list[str] = []

        def respond(http_request: httpx.Request) -> httpx.Response:
            if http_request.method == "GET":
                operations.append("receipt")
                return httpx.Response(
                    status_code=200,
                    json={
                        "request_id": request.request_id,
                        "target": (trigger if outcome == "wrong_victim" else victim).model_dump(mode="json"),
                        "mode": "sigkill",
                        "exited_pids": [42],
                    },
                )
            body = json.loads(http_request.content)
            if http_request.url.path.endswith("/inject-fault"):
                assert operations == ["inspect", "arm", "read"]
                assert body["expected_target"] == victim.model_dump(mode="json")
                assert body["request_id"] == request.request_id
                operations.append("inject")
                return httpx.Response(status_code=200)

            assert body["target"] == trigger.model_dump(mode="json")
            operation = body["command"]["operation"]
            operations.append(operation)
            if operation == "inspect":
                return httpx.Response(status_code=200, json="trigger-instance")
            hook_request = body["command"]["request"]
            assert hook_request["request_id"] == f"{request.request_id}:trigger"
            assert hook_request["action"] == "observe"
            if operation == "read" and outcome == "stale_trigger":
                return httpx.Response(status_code=412)
            status = "armed" if operation == "arm" else "fired"
            if operation == "read" and outcome == "expired":
                status = "expired"
            if operation == "read" and outcome == "wrong_instance":
                hook_request = {**hook_request, "instance_id": "replacement"}
            return httpx.Response(
                status_code=200,
                json={
                    "request": {**hook_request, "receipt_url": "http://control/receipt"},
                    "status": status,
                    "armed_at": 1.0,
                    "changed_at": 2.0,
                    "reached_at": 2.0,
                    "due_at": 2.0,
                    "weight_version": 37,
                    "update_id": None if outcome == "missing_update" else "update-37",
                },
            )

        client_type = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs)
        )

        if outcome == "hit":
            result = await form.execute(request)
            assert result["target"] == victim.model_dump(mode="json")
            assert result["hook_trigger"] == trigger.model_dump(mode="json")
            assert result["hook_hit"]["weight_version"] == 37
        else:
            error = (
                AssertionError
                if outcome == "changed_assignment"
                else (
                    httpx.HTTPStatusError
                    if outcome == "stale_trigger"
                    else RuntimeError if outcome == "expired" else ValueError
                )
            )
            with pytest.raises(error):
                await form.execute(request)

        assert operations == (
            ["inspect", "arm", "read", "inject", "receipt", "cancel"]
            if outcome in {"hit", "wrong_victim"}
            else ["inspect", "arm", "read", "cancel"]
        )

    @pytest.mark.parametrize(
        "outcome", ["confirmed", "lost_reply", "server_error", "stale", "cancelled", "wrong_receipt"]
    )
    async def test_arm_is_not_retried_and_cleanup_keeps_original_identity(
        self, monkeypatch: pytest.MonkeyPatch, outcome: str
    ) -> None:
        """Ambiguous arm replies require an effect receipt and every exit cancels the same incarnation."""
        form = HookFaultForm(
            base_url="http://control",
            failure_mode=FailureMode.SIGKILL,
            hook="trainer_before_all_gather",
            delay_ms=5,
        )
        identity = FaultTarget(cell_id="actor-7", sub_index=0, workers_hash="generation-0")
        request = SoakActionRequest(
            form_name=form.name, target=typed_cell("actor-7", "actor"), harms_cell=True, fault_target=identity
        )
        commands: list[dict] = []
        reads: list[str] = []
        receipt = {
            "request_id": request.request_id,
            "target": identity.model_dump(mode="json"),
            "mode": "sigkill",
            "exited_pids": [42],
        }

        def respond(http_request: httpx.Request) -> httpx.Response:
            if http_request.method == "GET":
                reads.append(http_request.url.path)
                if outcome == "cancelled":
                    raise asyncio.CancelledError
                if outcome == "wrong_receipt":
                    return httpx.Response(status_code=200, json={**receipt, "request_id": "another-request"})
                return httpx.Response(status_code=200, json=receipt)
            body = json.loads(http_request.content)
            commands.append(body)
            assert body["target"] == identity.model_dump(mode="json")
            operation = body["command"]["operation"]
            if operation == "inspect":
                return httpx.Response(status_code=200, json="worker-instance")
            if operation == "arm":
                if outcome == "lost_reply":
                    raise httpx.ReadTimeout("Reply lost", request=http_request)
                if outcome in {"server_error", "stale"}:
                    return httpx.Response(status_code=503 if outcome == "server_error" else 412)
            return httpx.Response(
                status_code=200,
                json={
                    "request": {**body["command"]["request"], "receipt_url": "http://control/receipt"},
                    "status": "armed" if operation == "arm" else "cancelled",
                    "armed_at": 1.0,
                    "changed_at": 1.0,
                },
            )

        client_type = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs)
        )

        if outcome == "stale":
            with pytest.raises(httpx.HTTPStatusError):
                await form.execute(request)
            assert reads == []
        elif outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await form.execute(request)
        elif outcome == "wrong_receipt":
            with pytest.raises(ValueError, match="does not match"):
                await form.execute(request)
        else:
            result = await form.execute(request)
            assert {key: result[key] for key in receipt} == receipt
            assert result["hook_request"]["instance_id"] == "worker-instance"
            assert result["hook_request"]["delay_ms"] == 5
            assert reads == [f"/api/v1/fault-receipts/{request.request_id}"]

        assert [body["command"]["operation"] for body in commands] == ["inspect", "arm", "cancel"]
        assert commands[1]["command"]["request"] == commands[2]["command"]["request"]
        assert commands[1]["command"]["request"]["request_id"] == request.request_id
