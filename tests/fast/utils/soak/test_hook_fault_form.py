import asyncio
import json

import httpx
import pytest
from tests.fast.utils.soak.utils import typed_cell
from tests.utils.soak.hook_fault_form import HookFaultForm
from tests.utils.soak.state import SoakActionRequest

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import FaultTarget


class TestHookFaultForm:
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
