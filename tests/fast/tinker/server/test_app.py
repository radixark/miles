"""One HTTP conversation over the real FastAPI app, plus the error mapping the
SDK's retry loop depends on (400/403/410/try_again)."""

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
    # the pinned SDK authenticates with X-API-Key, not Authorization
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


async def test_a_foreign_future_answers_403(client):
    created = (
        await client.post("/api/v1/create_model", json={"base_model": "base"}, headers=_headers("tenant-a"))
    ).json()
    response = await client.post(
        "/api/v1/retrieve_future", json={"request_id": created["request_id"]}, headers=_headers("tenant-b")
    )
    assert response.status_code == 403


async def test_an_unknown_future_answers_410(client):
    response = await client.post("/api/v1/retrieve_future", json={"request_id": "req-gone"}, headers=_headers())
    assert response.status_code == 410


async def test_a_failed_future_reports_the_category(client):
    client.service.backend.fail_next = RuntimeError("boom")
    created = (await client.post("/api/v1/create_model", json={"base_model": "base"}, headers=_headers())).json()
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
    created = (await client.post("/api/v1/create_model", json={"base_model": "base"}, headers=_headers())).json()
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
    assert (info["base_model"], info["is_lora"], info["lora_rank"]) == ("base", True, 32)
