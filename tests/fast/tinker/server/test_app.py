"""One HTTP conversation over the real FastAPI app, plus the error mapping the
SDK's retry loop depends on (400/403/410/try_again)."""

import asyncio

import httpx
import pytest
from tests.fast.tinker.harness import ADAM, make_service

from miles.tinker.server.app import build_app


@pytest.fixture
async def client():
    service = make_service()
    run_task = asyncio.create_task(service.run())
    transport = httpx.ASGITransport(app=build_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as http:
        http.service = service
        yield http
    for task in (run_task, getattr(service, "_sweep_task", None)):
        if task is not None:
            task.cancel()


def _headers(tenant: str = "tenant-a") -> dict:
    return {"Authorization": f"Bearer {tenant}"}


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

    created = (await client.post("/api/v1/create_model", json={"base_model": "base"}, headers=_headers())).json()
    assert (await _poll(client, created["request_id"]))["type"] == "create_model"
    model_id = created["model_id"]

    info = (await client.post("/api/v1/get_info", json={"model_id": model_id}, headers=_headers())).json()
    assert info["model_data"]["model_name"] == "base"

    fb = (await client.post("/api/v1/forward_backward", json=_fb_body(model_id, 1), headers=_headers())).json()
    fb_result = await _poll(client, fb["request_id"])
    assert [len(o["logprobs"]["data"]) for o in [fb_result["loss_fn_outputs"][0]]] == [3]

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
    body["forward_backward_input"]["data"][0]["loss_fn_inputs"]["target_tokens"] = [9, 9, 9]
    response = await client.post("/api/v1/forward_backward", json=body, headers=_headers())
    assert response.status_code == 400


async def test_a_foreign_promise_answers_403(client):
    created = (
        await client.post("/api/v1/create_model", json={"base_model": "base"}, headers=_headers("tenant-a"))
    ).json()
    response = await client.post(
        "/api/v1/retrieve_future", json={"request_id": created["request_id"]}, headers=_headers("tenant-b")
    )
    assert response.status_code == 403


async def test_an_unknown_promise_answers_410(client):
    response = await client.post("/api/v1/retrieve_future", json={"request_id": "req-gone"}, headers=_headers())
    assert response.status_code == 410


async def test_a_failed_promise_reports_the_category(client):
    client.service.backend.fail_next = RuntimeError("boom")
    created = (await client.post("/api/v1/create_model", json={"base_model": "base"}, headers=_headers())).json()
    body = await _poll(client, created["request_id"])
    assert (body["category"], "boom" in body["error"]) == ("internal", True)


async def test_capabilities_and_telemetry_shapes(client):
    capabilities = (await client.get("/api/v1/get_server_capabilities")).json()
    assert capabilities["supported_models"][0]["model_name"] == "base"
    assert (await client.post("/api/v1/telemetry", json={})).json() == {"status": "accepted"}


@pytest.mark.parametrize("route", ["get_info", "retrieve_future", "cancel_future", "optim_step"])
async def test_sdk_api_keys_isolate_models_and_promises(client, route):
    created = (
        await client.post("/api/v1/create_model", json={"base_model": "base"}, headers={"X-API-Key": "tenant-a"})
    ).json()
    payload = {
        "model_id": created["model_id"],
        "request_id": created["request_id"],
        "seq_id": 1,
        "adam_params": dict(ADAM),
    }
    response = await client.post(f"/api/v1/{route}", json=payload, headers={"X-API-Key": "tenant-b"})
    assert response.status_code == 403


async def test_sdk_and_bearer_credentials_identify_the_same_tenant(client):
    created = (
        await client.post("/api/v1/create_model", json={"base_model": "base"}, headers={"X-API-Key": "tenant-a"})
    ).json()
    response = await client.post("/api/v1/get_info", json={"model_id": created["model_id"]}, headers=_headers())
    assert response.status_code == 200


@pytest.mark.parametrize("bearer_tenant, status", [("tenant-a", 200), ("tenant-b", 400)])
async def test_dual_credentials_must_agree(client, bearer_tenant, status):
    response = await client.post(
        "/api/v1/create_session",
        json={},
        headers={"X-API-Key": "tenant-a", **_headers(bearer_tenant)},
    )
    assert response.status_code == status
    if status == 200:
        assert client.service.sessions[response.json()["session_id"]]["tenant"] == "tenant-a"


@pytest.mark.parametrize("header_name", ["X-API-Key", "Authorization"])
async def test_heartbeat_enforces_session_ownership(client, header_name):
    def headers(tenant):
        return {header_name: f"Bearer {tenant}" if header_name == "Authorization" else tenant}

    response = await client.post("/api/v1/create_session", json={}, headers=headers("tenant-a"))
    session_id = response.json()["session_id"]
    session = client.service.sessions[session_id]
    session["last_heartbeat"] -= 1
    previous = session["last_heartbeat"]

    foreign = await client.post(
        "/api/v1/session_heartbeat", json={"session_id": session_id}, headers=headers("tenant-b")
    )
    assert foreign.status_code == 403
    assert session["last_heartbeat"] == previous

    owner = await client.post(
        "/api/v1/session_heartbeat", json={"session_id": session_id}, headers=headers("tenant-a")
    )
    assert owner.status_code == 200
    assert session["last_heartbeat"] > previous
