"""The four /oai/sessions routes over build_app_with_collector: auth shapes, recording, ownership, error codes."""

import httpx
import pytest
from tests.fast.tinker.oai_fakes import OTHER_TENANT, SAMPLER, TENANT, FakeTokenizer, write_sampler

from miles.tinker.core.tinker_session_server import TrajectoryCollector
from miles.tinker.server.oai_routes import build_app_with_collector

OWNER = {"Authorization": f"Bearer {TENANT}"}
OTHER = {"X-API-Key": OTHER_TENANT}
DUMMY = {"Authorization": "Bearer dummy"}
CHAT = {"model": "openai/model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
HI = {"sequences": [{"tokens": [104, 105], "logprobs": [-0.1, -0.2], "stop_reason": "stop"}]}


@pytest.fixture
async def client(service, tmp_path):
    write_sampler(tmp_path, service.config.base_model)
    service.config.vocab_size = 1000
    collector = TrajectoryCollector(service, FakeTokenizer(), session_ttl_s=600.0, chat_template_kwargs=None)
    transport = httpx.ASGITransport(app=build_app_with_collector(service, collector))
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as http:
        http.service = service
        yield http


async def test_bind_then_dummy_key_chat_then_owner_reads(client):
    client.service.backend.fail_on["sample"] = HI
    bound = await client.post("/oai/sessions/s1", json={"model": SAMPLER}, headers=OWNER)
    assert bound.status_code == 200 and bound.json() == {"session_id": "s1", "model_path": SAMPLER}

    chat = await client.post("/oai/sessions/s1/v1/chat/completions", json=CHAT, headers=DUMMY)
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"] == "hi"
    assert client.service.backend.named("sample")[0]["lora_name"] == "m1@v0"

    assert (await client.get("/oai/sessions/s1", headers=OTHER)).status_code == 403
    assert (await client.get("/oai/sessions/s1")).status_code == 400  # GET needs the owner's key

    trajectory = (await client.get("/oai/sessions/s1", headers=OWNER)).json()
    assert trajectory["model_path"] == SAMPLER and len(trajectory["turns"]) == 1
    assert trajectory["turns"][0]["output_ids"] == [104, 105]

    assert (await client.delete("/oai/sessions/s1", headers=OWNER)).json() == {"session_id": "s1", "deleted": True}
    assert (await client.get("/oai/sessions/s1", headers=OWNER)).status_code == 404


async def test_unknown_session_without_bearer_is_404_and_auto_registers_with_bearer(client):
    assert (await client.post("/oai/sessions/nope/v1/chat/completions", json=CHAT)).status_code == 404
    registered = await client.post(
        "/oai/sessions/auto/v1/chat/completions", json={**CHAT, "model": SAMPLER}, headers=OWNER
    )
    assert registered.status_code == 200
    assert (await client.get("/oai/sessions/auto", headers=OWNER)).json()["model_path"] == SAMPLER


async def test_bad_inputs_are_400(client):
    assert (await client.post("/oai/sessions/s1", json={"model": SAMPLER})).status_code == 400  # no key
    assert (await client.post("/oai/sessions/s1", json={"model": "nonsense"}, headers=OWNER)).status_code == 400
    assert (await client.post("/oai/sessions/s1", json={"model": SAMPLER}, headers=OWNER)).status_code == 200
    empty = await client.post("/oai/sessions/s1/v1/chat/completions", json={"messages": []}, headers=DUMMY)
    assert empty.status_code == 400
    broken = await client.post("/oai/sessions/s1/v1/chat/completions", content=b"{", headers=DUMMY)
    assert broken.status_code == 400
    other_version = await client.post(
        "/oai/sessions/s1/v1/chat/completions",
        json={**CHAT, "model": "tinker://m1/sampler_weights/v9"},
        headers=DUMMY,
    )
    assert other_version.status_code == 400
    assert (await client.get("/oai/sessions/s1", headers=OWNER)).json()["turns"] == []


async def test_engine_failure_is_502_and_records_nothing(client):
    await client.post("/oai/sessions/s1", json={"model": SAMPLER}, headers=OWNER)
    client.service.backend.fail_on["sample"] = {"error": "engine aborted"}
    failed = await client.post("/oai/sessions/s1/v1/chat/completions", json=CHAT, headers=DUMMY)
    assert failed.status_code == 502 and failed.json() == {"error": "engine aborted"}
    assert (await client.get("/oai/sessions/s1", headers=OWNER)).json()["turns"] == []


async def test_tinker_routes_are_untouched(client):
    assert (await client.get("/api/v1/healthz")).json() == {"status": "ok"}
    capabilities = (await client.post("/api/v1/get_server_capabilities", json={})).json()
    assert capabilities["supported_models"][0]["model_name"] == client.service.config.base_model
