"""Session chats whose replay outputs come back through the SGLang output store.

Drives the real session routes with a fake backend that answers like an SGLang
engine with ``--output-store-backend mooncake``: R3 rows travel as an
``output_store_ref`` to a fake Mooncake store instead of inline base64.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pybase64
import pytest
import pytest_asyncio
from fastapi import FastAPI
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.generate_utils.output_store import OUTPUT_STORE_REF_KEY
from miles.rollout.session import sessions
from miles.rollout.session.replay_reads import ReplayReader
from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS,
    COMPUTED_FIELDS_V2,
    decode_samples_and_merge_input_sample,
)
from miles.utils import object_store
from miles.utils.object_store import MooncakeObjectStore
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

pytestmark = pytest.mark.asyncio
NUM_LAYERS, TOPK = 2, 3
USER = {"role": "user", "content": "hello"}
ASSISTANT = {"role": "assistant", "content": "answer"}
TOOL = {"role": "tool", "content": "result", "tool_call_id": "call_1"}


def _r3_rows(start: int, end: int) -> np.ndarray:
    """Rows [start, end) of one trajectory's R3, distinct per row so misplaced patches show."""
    rows = np.arange(start, end, dtype=np.int32)[:, None, None] + 1
    return rows * 100 + np.arange(NUM_LAYERS, dtype=np.int32)[:, None] * 10 + np.arange(TOPK, dtype=np.int32)


class _KeyedStore(MooncakeObjectStore):
    """Mooncake boundary holding one bundle per handle id, as the engine's puts left them."""

    def __init__(self):
        self.bundles: dict[int, dict] = {}
        self.reads = 0
        self.removed: list[int] = []

    def get(self, ref):
        self.reads += 1
        return object_store.ObjectStoreGetResult(value=self.bundles[ref.payload["id"]], release_fn=lambda value: None)

    def remove(self, ref):
        self.removed.append(ref.payload["id"])
        del self.bundles[ref.payload["id"]]


class _Backend:
    def __init__(self, tokenizer, store: _KeyedStore, *, via_store: bool):
        self.tokenizer = tokenizer
        self.store = store
        self.via_store = via_store
        self.requests = []
        self.gate: asyncio.Event | None = None

    async def do_proxy(self, request, path, *, body, headers):
        payload = json.loads(body)
        self.requests.append(payload)
        if self.gate is not None:
            await self.gate.wait()
        render = dict(tokenize=False, enable_thinking=False)
        prompt = self.tokenizer.apply_chat_template(payload["messages"], add_generation_prompt=True, **render)
        complete = self.tokenizer.apply_chat_template(payload["messages"] + [ASSISTANT], **render)
        output_ids = self.tokenizer.encode(complete[len(prompt) :], add_special_tokens=False)
        rows = _r3_rows(payload.get("routed_experts_start_len", 0), len(payload["input_ids"]) + len(output_ids) - 1)
        meta_info = {"completion_tokens": len(output_ids), "output_token_logprobs": [[-0.1, t] for t in output_ids]}
        if self.via_store:
            handle = {"type": "mooncake_dataproto_ref", "id": len(self.requests)}
            self.store.bundles[handle["id"]] = {"routed_experts": [rows.tolist()]}
            meta_info[OUTPUT_STORE_REF_KEY] = {
                "handle": handle,
                "fields": {"routed_experts": {"dtype": "int32", "shape": list(rows.shape)}},
            }
        else:
            meta_info["routed_experts"] = pybase64.b64encode(rows.tobytes()).decode("ascii")
        response = {
            "id": f"response-{len(self.requests)}",
            "choices": [{"index": 0, "message": ASSISTANT, "finish_reason": "stop", "meta_info": meta_info}],
        }
        return {
            "request_body": body,
            "response_body": json.dumps(response).encode(),
            "status_code": 200,
            "headers": {"content-type": "application/json"},
        }


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer("Qwen/Qwen3-0.6B", trust_remote_code=True)


async def _serve(tokenizer, monkeypatch, version, *, via_store=True):
    monkeypatch.setattr(sessions, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    store = _KeyedStore()
    backend = _Backend(tokenizer, store, via_store=via_store)
    config = make_session_server_config(
        hf_checkpoint="Qwen/Qwen3-0.6B",
        apply_chat_template_kwargs={"enable_thinking": False},
        use_session_server=version,
        use_rollout_routing_replay=True,
        num_layers=NUM_LAYERS,
        moe_router_topk=TOPK,
        sglang_output_store_backend="mooncake" if via_store else "none",
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    reader = ReplayReader(store) if via_store else None
    app = FastAPI()
    sessions.setup_session_routes(app, backend, config, use_addition_r3=True, replay_reader=reader)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://session") as client:
        yield SimpleNamespace(client=client, backend=backend, store=store, version=version)
    if reader is not None:
        reader.close()


@pytest_asyncio.fixture(params=[True, "v2"], ids=["v1", "v2"])
async def env(request, tokenizer, monkeypatch):
    async for value in _serve(tokenizer, monkeypatch, request.param):
        yield value


async def _two_turn_session(env) -> str:
    sid = (await env.client.post("/sessions")).json()["session_id"]
    for messages in ([USER], [USER, ASSISTANT, TOOL]):
        response = await env.client.post(f"/sessions/{sid}/v1/chat/completions", json={"messages": messages})
        assert response.status_code == 200, response.text
        assert "meta_info" not in response.json()["choices"][0]
    return sid


async def _samples(env, sid, **body) -> list[Sample]:
    response = await env.client.post(f"/sessions/{sid}/samples", json=body)
    assert response.status_code == 200, response.text
    fields = COMPUTED_FIELDS_V2 if env.version == "v2" else COMPUTED_FIELDS
    return decode_samples_and_merge_input_sample(response.content, Sample(), fields=fields).samples


async def test_patches_read_from_the_store_rebuild_the_trajectory_r3(env):
    sid = await _two_turn_session(env)

    (sample,) = await _samples(env, sid)

    assert all(request["return_outputs_via_store"] is True for request in env.backend.requests)
    np.testing.assert_array_equal(sample.rollout_routed_experts, _r3_rows(0, len(sample.tokens) - 1))
    assert env.store.bundles == {}


async def test_a_second_collect_reuses_the_arrays_without_reading_again(env):
    sid = await _two_turn_session(env)
    (full,) = await _samples(env, sid)
    reads = env.store.reads

    (truncated,) = await _samples(env, sid, max_seq_len=len(full.tokens) - 1)

    assert env.store.reads == reads
    np.testing.assert_array_equal(truncated.rollout_routed_experts, _r3_rows(0, len(full.tokens) - 2))


async def test_a_response_discarded_by_a_concurrent_delete_still_has_its_bundle_removed(env):
    sid = (await env.client.post("/sessions")).json()["session_id"]
    env.backend.gate = asyncio.Event()
    chat = asyncio.create_task(env.client.post(f"/sessions/{sid}/v1/chat/completions", json={"messages": [USER]}))
    while not env.backend.requests:
        await asyncio.sleep(0)

    assert (await env.client.delete(f"/sessions/{sid}")).status_code == 204
    env.backend.gate.set()
    assert (await chat).status_code == 200
    for _ in range(100):
        if env.store.removed:
            break
        await asyncio.sleep(0.01)

    assert env.store.removed == [1]


async def test_inline_and_store_sessions_assemble_the_same_sample(tokenizer, monkeypatch):
    samples = []
    for via_store in (False, True):
        async for env in _serve(tokenizer, monkeypatch, "v2", via_store=via_store):
            (sample,) = await _samples(env, await _two_turn_session(env))
            samples.append(sample)

    inline, via_store = samples
    np.testing.assert_array_equal(via_store.rollout_routed_experts, inline.rollout_routed_experts)
    assert (via_store.tokens, via_store.rollout_log_probs) == (inline.tokens, inline.rollout_log_probs)
