import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session import sessions
from miles.rollout.session.core import SessionCore
from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS,
    COMPUTED_FIELDS_V2,
    decode_samples_and_merge_input_sample,
)
from miles.rollout.session.v2.session_state import SessionRegistryV2
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

pytestmark = pytest.mark.asyncio
REPLAY_FIELDS = ("return_sampling_mask", "return_routed_experts", "return_indexer_topk")
USER = {"role": "user", "content": "hello"}
ASSISTANT = {"role": "assistant", "content": "answer"}
TOOL = {"role": "tool", "content": "result", "tool_call_id": "call_1"}


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer("Qwen/Qwen3-0.6B", trust_remote_code=True)


class _Backend:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.requests = []

    async def do_proxy(self, request, path, *, body, headers):
        payload = json.loads(body)
        self.requests.append(payload)
        await asyncio.sleep(0)
        render = dict(tokenize=False, enable_thinking=False)
        prompt = self.tokenizer.apply_chat_template(payload["messages"], add_generation_prompt=True, **render)
        complete = self.tokenizer.apply_chat_template(payload["messages"] + [ASSISTANT], **render)
        assert complete.startswith(prompt)
        output_ids = self.tokenizer.encode(complete[len(prompt) :], add_special_tokens=False)
        response = {
            "id": f"response-{len(self.requests)}",
            "choices": [
                {
                    "index": 0,
                    "message": ASSISTANT,
                    "finish_reason": "stop",
                    "meta_info": {
                        "completion_tokens": len(output_ids),
                        "output_token_logprobs": [[-0.1, token] for token in output_ids],
                    },
                }
            ],
        }
        return {
            "request_body": body,
            "response_body": json.dumps(response).encode(),
            "status_code": 200,
            "headers": {"content-type": "application/json"},
        }


async def _serve_env(tokenizer, monkeypatch, version, *, use_sampling_support_replay=False):
    monkeypatch.setattr(sessions, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    backend = _Backend(tokenizer)
    config = make_session_server_config(
        hf_checkpoint="Qwen/Qwen3-0.6B",
        apply_chat_template_kwargs={"enable_thinking": False},
        use_session_server=version,
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=True,
        use_sampling_support_replay=use_sampling_support_replay,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_retries",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    app = FastAPI()
    sessions.setup_session_routes(app, backend, config, use_addition_r3=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://session") as client:
        yield SimpleNamespace(client=client, backend=backend, version=version)


@pytest_asyncio.fixture(params=[True, "v2"], ids=["v1", "v2"])
async def env(request, tokenizer, monkeypatch):
    async for value in _serve_env(tokenizer, monkeypatch, request.param):
        yield value


@pytest_asyncio.fixture(params=[True, "v2"], ids=["v1", "v2"])
async def replay_env(request, tokenizer, monkeypatch):
    async for value in _serve_env(
        tokenizer,
        monkeypatch,
        request.param,
        use_sampling_support_replay=True,
    ):
        yield value


async def _create(env, body=b""):
    response = await env.client.post("/sessions", content=body)
    assert response.status_code == 200, response.text
    assert set(response.json()) == {"session_id"}
    return response.json()["session_id"]


async def _chat(env, sid, messages, **kwargs):
    response = await env.client.post(f"/sessions/{sid}/v1/chat/completions", json={"messages": messages, **kwargs})
    assert response.status_code == 200, response.text
    return response


@pytest.mark.parametrize("body", [b"", b"{}", b'{"evaluation": false}', b'{"evaluation": true}'])
async def test_creation_selects_policy_and_defaults_to_training(env, body):
    sid = await _create(env, body)
    await _chat(env, sid, [USER])
    wire = env.backend.requests[-1]
    evaluation = body == b'{"evaluation": true}'
    assert wire["return_routed_experts"] is (not evaluation)
    assert wire["return_indexer_topk"] is (not evaluation)
    if evaluation:
        assert wire["return_sampling_mask"] is False
        assert "routed_experts_start_len" not in wire
    else:
        assert "return_sampling_mask" not in wire
        assert wire["routed_experts_start_len"] == 0


@pytest.mark.parametrize(
    "body",
    [
        b"{",
        b"[]",
        b"null",
        b'{"evaluation": null}',
        b'{"evaluation": 0}',
        b'{"evaluation": 1}',
        b'{"evaluation": "false"}',
        b'{"evalution": true}',
        b'{"unexpected": 1}',
        b'{"temperature": "0.6"}',
        b'{"top_p": [0.9]}',
        b'{"top_k": 20.5}',
        b'{"top_k": true}',
    ],
)
async def test_invalid_creation_is_rejected_before_allocation(env, body, monkeypatch):
    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("invalid creation must not allocate a session")

    monkeypatch.setattr(sessions.SessionRegistry, "create_session", unexpected_allocation)
    monkeypatch.setattr(SessionRegistryV2, "create_session", unexpected_allocation)
    response = await env.client.post("/sessions", content=body)
    assert response.status_code == 400
    assert env.backend.requests == []


async def test_eval_policy_survives_continuation_and_retry_and_collects_without_replay(env):
    sid = await _create(env, b'{"evaluation": true}')
    turns = [([USER], 0.0), ([USER, ASSISTANT, TOOL], 0.7), ([USER, ASSISTANT, {**TOOL, "content": "retry"}], 1.2)]
    for messages, temperature in turns:
        await _chat(
            env,
            sid,
            messages,
            temperature=temperature,
            top_p=0.8,
            top_k=-1,
            routed_experts_start_len=999,
            **dict.fromkeys(REPLAY_FIELDS, True),
        )
        wire = env.backend.requests[-1]
        assert all(wire[field] is False for field in REPLAY_FIELDS)
        assert "routed_experts_start_len" not in wire
        assert wire["temperature"] == temperature
        assert wire["top_p"] == 0.8 and wire["top_k"] == -1
        assert wire["logprobs"] is True and wire["return_meta_info"] is True
        assert wire["no_stop_trim"] is False
    response = await env.client.post(f"/sessions/{sid}/samples", json={})
    assert response.status_code == 200, response.text
    fields = COMPUTED_FIELDS_V2 if env.version == "v2" else COMPUTED_FIELDS
    reply = decode_samples_and_merge_input_sample(response.content, Sample(), fields=fields)
    assert reply.samples
    assert all(
        sample.rollout_routed_experts is None and sample.rollout_indexer_topk is None for sample in reply.samples
    )
    assert (await env.client.delete(f"/sessions/{sid}")).status_code == 204


async def test_concurrent_train_and_eval_do_not_share_policy(env):
    train, evaluation = await asyncio.gather(_create(env), _create(env, b'{"evaluation": true}'))
    await asyncio.gather(_chat(env, train, [USER]), _chat(env, evaluation, [USER]))
    assert sorted(request["return_routed_experts"] for request in env.backend.requests) == [False, True]
    assert sorted(request["return_indexer_topk"] for request in env.backend.requests) == [False, True]


SAMPLING = {"temperature": 0.6, "top_p": 0.9, "top_k": 20}


@pytest.mark.parametrize("evaluation", [False, True])
async def test_creation_sampling_defaults_fill_only_omitted_fields(replay_env, evaluation):
    sid = await _create(replay_env, json.dumps({**SAMPLING, "evaluation": evaluation}).encode())
    await _chat(replay_env, sid, [USER])
    temperature = 0.1 if evaluation else SAMPLING["temperature"]
    await _chat(replay_env, sid, [USER, ASSISTANT, TOOL], temperature=temperature, top_p=None, top_k=10)
    omitted, explicit = replay_env.backend.requests[-2:]
    assert {key: omitted[key] for key in SAMPLING} == SAMPLING
    assert {key: explicit[key] for key in SAMPLING} == {**SAMPLING, "temperature": temperature, "top_k": 10}
    assert omitted["return_sampling_mask"] is (not evaluation)
    assert explicit["return_sampling_mask"] is (not evaluation)


@pytest.mark.parametrize("saved,requested", [(0.6, 0.1), (0.6, 0.0), (0.0, 0.6)])
async def test_training_temperature_mismatch_is_rejected_before_forwarding(replay_env, saved, requested):
    sid = await _create(replay_env, json.dumps({**SAMPLING, "temperature": saved}).encode())
    turns = [[USER], [USER, ASSISTANT, TOOL], [USER, ASSISTANT, TOOL]]
    for messages in turns:
        request_count = len(replay_env.backend.requests)
        response = await replay_env.client.post(
            f"/sessions/{sid}/v1/chat/completions",
            json={"messages": messages, "temperature": requested},
        )
        assert response.status_code == 400
        assert response.json()["error"] == (
            f"temperature={requested!r} does not match the training session temperature={saved!r}"
        )
        assert len(replay_env.backend.requests) == request_count
        await _chat(replay_env, sid, messages, temperature=saved, top_p=0.5, top_k=10)
        wire = replay_env.backend.requests[-1]
        assert (wire["temperature"], wire["top_p"], wire["top_k"]) == (saved, 0.5, 10)
        assert wire["return_sampling_mask"] is True


async def test_training_null_temperature_uses_registered_value(replay_env):
    sid = await _create(replay_env, json.dumps(SAMPLING).encode())
    await _chat(replay_env, sid, [USER], temperature=None)
    assert replay_env.backend.requests[-1]["temperature"] == SAMPLING["temperature"]


async def test_creation_without_sampling_defaults_leaves_omitted_fields_unset(env):
    sid = await _create(env)
    await _chat(env, sid, [USER])
    assert not set(SAMPLING) & set(env.backend.requests[-1])
    await _chat(env, sid, [USER, ASSISTANT, TOOL], temperature=0.4)
    assert env.backend.requests[-1]["temperature"] == 0.4


async def test_an_integer_temperature_is_accepted_as_a_float_default(env):
    sid = await _create(env, b'{"temperature": 0}')
    await _chat(env, sid, [USER])
    assert env.backend.requests[-1]["temperature"] == 0
    assert "top_p" not in env.backend.requests[-1]


async def test_an_integral_float_top_k_is_stored_as_an_int(env):
    """An eval dataset YAML can spell top_k as 40.0; the engine must still receive an integer."""
    sid = await _create(env, b'{"evaluation": true, "top_k": 40.0}')
    await _chat(env, sid, [USER])
    top_k = env.backend.requests[-1]["top_k"]
    assert top_k == 40 and isinstance(top_k, int)


async def test_concurrent_sessions_keep_their_own_sampling_defaults(env):
    first, second = await asyncio.gather(_create(env, b'{"temperature": 0.2}'), _create(env, b'{"temperature": 0.8}'))
    await asyncio.gather(_chat(env, first, [USER]), _chat(env, second, [USER]))
    assert sorted(request["temperature"] for request in env.backend.requests) == [0.2, 0.8]


@pytest.mark.parametrize("field", ["input_ids", "logprob_start_len", "lora_path"])
async def test_eval_retains_tito_control_validation(env, field):
    sid = await _create(env, b'{"evaluation": true}')
    response = await env.client.post(f"/sessions/{sid}/v1/chat/completions", json={"messages": [USER], field: 1})
    assert response.status_code == 400
    assert env.backend.requests == []


async def test_eval_does_not_run_additional_r3_prefix_assertion():
    core = SessionCore(None, None, make_session_server_config(), use_addition_r3=True)
    body = {"return_routed_experts": False}
    core._maybe_request_addition_r3(body, [1, 2, 3, 4], [99])
    assert "routed_experts_start_len" not in body
    with pytest.raises(AssertionError, match="additional R3 requires"):
        core._maybe_request_addition_r3({"return_routed_experts": True}, [1, 2, 3, 4], [99])
