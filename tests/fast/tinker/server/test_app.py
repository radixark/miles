"""HTTP responses preserve the SDK conversation and retry/error contract."""

import asyncio

import httpx
import pytest
from tests.fast.tinker.harness import ADAM

from miles.tinker.server.app import build_app


@pytest.fixture
async def client(service):
    transport = httpx.ASGITransport(app=build_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as http:
        http.service = service
        yield http


def _headers(tenant: str = "tenant-a") -> dict:
    return {"X-API-Key": tenant}


async def _poll(client, request_id: str, tenant: str = "tenant-a") -> dict:
    for _ in range(400):
        response = await client.post(
            "/api/v1/retrieve_future", json={"request_id": request_id}, headers=_headers(tenant)
        )
        body = response.json()
        if body.get("type") != "try_again":
            return body
        await asyncio.sleep(0.005)
    raise AssertionError(f"{request_id} never settled")


async def _model_body(client, tenant: str = "tenant-a", **extra) -> dict:
    session = (await client.post("/api/v1/create_session", json={}, headers=_headers(tenant))).json()
    return {"base_model": "base", "session_id": session["session_id"], "model_seq_id": 1, **extra}


def _fb_body(model_id: str, seq_id: int) -> dict:
    return {
        "model_id": model_id,
        "seq_id": seq_id,
        "forward_backward_input": {
            "data": [
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                    "loss_fn_inputs": {"target_tokens": [2, 3, 4], "weights": [1.0, 1.0, 1.0]},
                }
            ],
            "loss_fn": "cross_entropy",
        },
    }


async def test_the_training_conversation(client):
    session = await client.post("/api/v1/create_session", json={}, headers=_headers())
    assert session.json()["session_id"].startswith("session-")

    body = {"base_model": "base", "session_id": session.json()["session_id"], "model_seq_id": 1}
    created = (await client.post("/api/v1/create_model", json=body, headers=_headers())).json()
    assert (await _poll(client, created["request_id"]))["type"] == "create_model"
    model_id = created["model_id"]

    info = (await client.post("/api/v1/get_info", json={"model_id": model_id}, headers=_headers())).json()
    assert info["model_data"]["model_name"] == "base"

    fb = (await client.post("/api/v1/forward_backward", json=_fb_body(model_id, 1), headers=_headers())).json()
    fb_result = await _poll(client, fb["request_id"])
    assert len(fb_result["loss_fn_outputs"][0]["logprobs"]["data"]) == 3

    optim = (
        await client.post(
            "/api/v1/optim_step",
            json={"model_id": model_id, "seq_id": 2, "adam_params": dict(ADAM)},
            headers=_headers(),
        )
    ).json()
    optim_result = await _poll(client, optim["request_id"])
    assert "grad_norm" in optim_result["metrics"]

    sampler = (
        await client.post(
            "/api/v1/save_weights_for_sampler", json={"model_id": model_id, "seq_id": 3}, headers=_headers()
        )
    ).json()
    sampler_path = (await _poll(client, sampler["request_id"]))["path"]

    sample = (
        await client.post(
            "/api/v1/asample",
            json={
                "model_path": sampler_path,
                "num_samples": 2,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
                "sampling_params": {"max_tokens": 4},
            },
            headers=_headers(),
        )
    ).json()
    assert len(sample["sample_sequence_ids"]) == 2, "asample must answer an UntypedAPIFuture"
    assert len((await _poll(client, sample["request_id"]))["sequences"]) == 2


async def test_bad_input_answers_400(client):
    body = _fb_body("model-missing", 1)
    body["forward_backward_input"]["data"][0]["loss_fn_inputs"]["target_tokens"] = [9, 9]
    response = await client.post("/api/v1/forward_backward", json=body, headers=_headers())
    assert response.status_code == 400


async def test_a_foreign_future_answers_403(client):
    created = (
        await client.post("/api/v1/create_model", json=await _model_body(client), headers=_headers("tenant-a"))
    ).json()
    response = await client.post(
        "/api/v1/retrieve_future", json={"request_id": created["request_id"]}, headers=_headers("tenant-b")
    )
    assert response.status_code == 403


async def test_an_unknown_future_answers_410(client):
    response = await client.post("/api/v1/retrieve_future", json={"request_id": "req-gone"}, headers=_headers())
    assert response.status_code == 410


async def test_a_failed_future_reports_the_category(client):
    body = await _model_body(client)
    client.service.backend.fail_next = {"error": "boom"}
    created = (await client.post("/api/v1/create_model", json=body, headers=_headers())).json()
    body = await _poll(client, created["request_id"])
    assert (body["category"], "boom" in body["error"]) == ("server", True)


async def test_capabilities_and_telemetry_shapes(client):
    capabilities = (await client.get("/api/v1/get_server_capabilities")).json()
    assert capabilities["supported_models"][0]["model_name"] == "base"
    assert (await client.post("/api/v1/telemetry", json={})).json() == {"status": "accepted"}


async def test_a_request_without_an_api_key_is_rejected(client):
    response = await client.post("/api/v1/create_session", json={})
    assert response.status_code == 400
    assert "X-API-Key" in response.json()["error"]


async def test_a_bearer_authorization_still_authenticates(client):
    response = await client.post("/api/v1/create_session", json={}, headers={"Authorization": "Bearer tenant-b"})
    assert response.status_code == 200


async def test_weights_info_answers_the_sdk_resume_probe(client):
    created = (
        await client.post(
            "/api/v1/create_model",
            json=await _model_body(client, lora_config={"rank": 8}),
            headers=_headers(),
        )
    ).json()
    await _poll(client, created["request_id"])
    saved = (
        await client.post(
            "/api/v1/save_weights",
            json={"model_id": created["model_id"], "seq_id": 1, "path": "ck", "overwrite": False},
            headers=_headers(),
        )
    ).json()
    path = (await _poll(client, saved["request_id"]))["path"]
    info = (await client.post("/api/v1/weights_info", json={"tinker_path": path}, headers=_headers())).json()
    assert (info["base_model"], info["is_lora"], info["lora_rank"]) == ("base", True, 8)


async def test_retrieve_long_polls_until_settlement(client):
    future = client.service.futures.create("model", "tenant-a")
    poll = asyncio.create_task(
        client.post("/api/v1/retrieve_future", json={"request_id": future.request_id}, headers=_headers())
    )
    await asyncio.sleep(0.05)
    assert not poll.done(), "a pending future must hold the poll open instead of answering try_again"
    client.service.futures.resolve(future.request_id, {"op": "optim_step", "metrics": {"grad_norm": 1.0}})
    response = await asyncio.wait_for(poll, timeout=2)
    assert response.json() == {"type": "optim_step", "metrics": {"grad_norm": 1.0}}


async def test_sdk_close_frees_the_slot_without_waiting_for_the_lease(client):
    """SDK ServiceClient.close(status) POSTs this body and expects 204; the lease sweep must not be needed."""
    body = await _model_body(client)
    session_id = body["session_id"]
    created = (await client.post("/api/v1/create_model", json=body, headers=_headers())).json()
    await _poll(client, created["request_id"])
    model_id = created["model_id"]
    slot = client.service.models[model_id].slot
    sampler = (
        await client.post(
            "/api/v1/create_sampling_session",
            json={"session_id": session_id, "sampling_session_seq_id": 0, "base_model": "base"},
            headers=_headers(),
        )
    ).json()["sampling_session_id"]

    finished = await client.post(
        f"/api/v1/sessions/{session_id}/finish",
        json={"reason": {"type": "success"}, "detail": None},
        headers=_headers(),
    )

    assert finished.status_code == 204
    assert slot in client.service.free_slots
    assert client.service.backend.named("unload_slot") == [{"slot": slot}]
    assert (await client.get(f"/api/v1/samplers/{sampler}", headers=_headers())).status_code == 400
    info = await client.post("/api/v1/get_info", json={"model_id": model_id}, headers=_headers())
    assert "session finished (success)" in info.json()["error"]


async def test_finishing_another_tenants_session_is_refused(client):
    session_id = (await _model_body(client))["session_id"]

    finished = await client.post(
        f"/api/v1/sessions/{session_id}/finish",
        json={"reason": {"type": "success"}, "detail": None},
        headers=_headers("tenant-b"),
    )

    assert finished.status_code == 400
    assert session_id in client.service.sessions


async def test_a_retried_finish_is_accepted_and_a_finish_after_lease_expiry_is_refused(client):
    """SDK close() retries on timeouts, so a repeat finish must not fail; a lost lease must not pass silently."""
    finish_body = {"reason": {"type": "success"}, "detail": None}
    finished = (await _model_body(client))["session_id"]
    expired = (await _model_body(client))["session_id"]
    await client.post(f"/api/v1/sessions/{finished}/finish", json=finish_body, headers=_headers())
    client.service.sessions[expired].last_heartbeat -= client.service.config.lease_timeout_s + 1
    await client.service._expire_sessions()

    retried = await client.post(f"/api/v1/sessions/{finished}/finish", json=finish_body, headers=_headers())
    foreign = await client.post(f"/api/v1/sessions/{finished}/finish", json=finish_body, headers=_headers("tenant-b"))
    late = await client.post(f"/api/v1/sessions/{expired}/finish", json=finish_body, headers=_headers())

    assert retried.status_code == 204
    assert foreign.status_code == 400
    assert late.status_code == 410


async def test_ended_sessions_are_forgotten_one_lease_after_they_end(client):
    """Retry records age out with the lease instead of being capped by the slot count."""
    service = client.service
    sessions = [(await _model_body(client))["session_id"] for _ in range(4 * service.config.n_slots + 1)]
    for session_id in sessions:
        service.sessions[session_id].last_heartbeat -= service.config.lease_timeout_s + 1
    await service._expire_sessions()
    assert set(sessions) <= set(service._ended_sessions)

    for ended in list(service._ended_sessions):
        service._ended_sessions[ended] = service._ended_sessions[ended]._replace(
            ended_at=service._ended_sessions[ended].ended_at - service.config.lease_timeout_s
        )
    await service._expire_sessions()
    assert service._ended_sessions == {}
