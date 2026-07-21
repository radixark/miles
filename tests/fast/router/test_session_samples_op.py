"""Samples-op tests: golden assembled Samples plus the op-level error/route contracts.

Drives `SessionCore.collect_samples` in-process against a real tokenizer (the
`test_sessions.py` precedent), with records injected via the registry — the
broken-chain and R3 fixtures cannot be produced through the chat path. The
HTTP surface (route registration order, 404 mapping) is exercised through the
real `setup_session_routes` app with a `TestClient`.

The golden tests assert the exact `Sample` field values derivable from the
two-turn records fixture — through `collect_samples` → `decode_samples_reply`
overlay → the driver-side metadata application `agentic_tool_call.generate`
performs — including the template-field overlay and the metadata application
order (agent metadata overrides the input's keys; session metadata, applied
last, overrides the agent's).
"""

import json
import uuid
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.fast.rollout.session.test_samples import _make_record

from miles.rollout.session.core import SessionCore
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.samples_codec import decode_samples_reply
from miles.rollout.session.sessions import setup_session_routes
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

NUM_LAYERS = 3
TOPK = 2

_ARGS = SimpleNamespace(
    miles_router_timeout=30,
    hf_checkpoint="Qwen/Qwen3-0.6B",
    chat_template_path=None,
    apply_chat_template_kwargs={"enable_thinking": False},
    tito_model="default",
    tito_allowed_append_roles=["tool"],
    session_server_instance_id=uuid.uuid4().hex,
    num_layers=NUM_LAYERS,
    moe_router_topk=TOPK,
)


class _UnusedBackend:
    """collect_samples never proxies; any backend call is a test bug."""

    async def do_proxy(self, *args, **kwargs):
        raise AssertionError("collect_samples must not touch the proxy backend")


def _build_core() -> SessionCore:
    # Mirrors setup_session_routes (sessions.py): tokenizer + registry + core.
    tokenizer = load_tokenizer(
        _ARGS.hf_checkpoint, chat_template_path=_ARGS.chat_template_path, trust_remote_code=True
    )
    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=_ARGS.tito_model,
        chat_template_kwargs=_ARGS.apply_chat_template_kwargs,
        allowed_append_roles=_ARGS.tito_allowed_append_roles,
    )
    registry = SessionRegistry(_ARGS, tokenizer, tito_tokenizer=tito_tokenizer)
    return SessionCore(_UnusedBackend(), registry, _ARGS, _ARGS.session_server_instance_id)


@pytest.fixture(scope="module")
def core():
    return _build_core()


# ── fixtures: a two-turn trajectory with R3 / cache stats / weight versions ──


def _r3_b64(num_tokens: int, seed: int) -> str:
    arr = np.arange(seed, seed + num_tokens * NUM_LAYERS * TOPK, dtype=np.int32)
    return pybase64.b64encode(arr.tobytes()).decode("ascii")


def _two_turn_records():
    # R3 buffer length per record = (len(prompt) + len(output) - 1) * layers * topk.
    return [
        _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            output_log_probs=[-0.125, -0.25],
            cached_tokens=0,
            prompt_tokens=3,
            weight_version="w1",
            routed_experts=_r3_b64(4, seed=0),
        ),
        _make_record(
            prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
            output_token_ids=[30, 31],
            output_log_probs=[-0.5, -1.0],
            cached_tokens=5,
            prompt_tokens=7,
            weight_version="w2",
            routed_experts=_r3_b64(8, seed=100),
        ),
    ]


_ACCUMULATED = [1, 2, 3, 10, 11, 20, 21, 30, 31]


def _input_sample() -> Sample:
    return Sample(
        group_index=4,
        index=9,
        prompt=[{"role": "user", "content": "hi"}],
        label="lbl",
        reward=2.5,
        metadata={"task": "t1", "shared_key": "from-input"},
        routing_key="routing-sid",
        train_metadata={"loss": "ppo"},
        generate_function_path="gen.fn",
    )


# Overlapping keys lock the application order: agent overrides the input's
# shared_key; session_metadata (applied last) overrides the agent's
# max_trim_tokens plant.
_AGENT_METADATA = {"shared_key": "from-agent", "agent_only": 1, "max_trim_tokens": "agent-plant"}


async def _make_session(core, records, accumulated) -> str:
    response = await core.create_session()
    sid = json.loads(response.body)["session_id"]
    session = core.registry.sessions[sid]
    for record in records:
        session.append_record(record)
    if accumulated is not None:
        session.trajectory_token_ids.append(list(accumulated))
    return sid


async def _collect_via_op(core, sid, *, multi_samples=False, max_seq_len=None):
    response = await core.collect_samples(sid, multi_samples=multi_samples, max_seq_len=max_seq_len)
    return response.status_code, response.body


def _new_pipeline(payload, input_sample, *, multi_samples):
    """What collect_samples() does after the cutover: overlay + driver-side metadata."""
    reply = decode_samples_reply(payload, input_sample)
    samples = reply.samples
    for s in samples:
        s.metadata.update(_AGENT_METADATA)
    if samples:
        if not multi_samples:
            (merged,) = samples
            merged.metadata.update(reply.session_metadata)
        else:
            samples[-1].metadata.update(reply.session_metadata)
    return samples, reply


# ── golden assembly: exact expected Samples for the two-turn fixture ──


def _expected_r3(seed: int, num_tokens: int):
    return np.arange(seed, seed + num_tokens * NUM_LAYERS * TOPK, dtype=np.int32).reshape(num_tokens, NUM_LAYERS, TOPK)


async def test_assembled_samples_golden_multi(core):
    """multi_samples=True, no truncation: per-turn Samples carry exactly the
    values derivable from the records fixture, template fields come from the
    driver's input sample, and the metadata application order holds."""
    sid = await _make_session(core, _two_turn_records(), _ACCUMULATED)
    status, payload = await _collect_via_op(core, sid, multi_samples=True)
    assert status == 200
    samples, reply = _new_pipeline(payload, _input_sample(), multi_samples=True)
    assert reply.empty_reason is None
    tokenizer = core.registry.tokenizer

    s1, s2 = samples
    assert s1.tokens == [1, 2, 3, 10, 11]
    assert s1.response == tokenizer.decode([10, 11])
    assert s1.response_length == 2
    assert s1.loss_mask == [1, 1]
    assert s1.rollout_log_probs == [-0.125, -0.25]
    assert s1.status == Sample.Status.COMPLETED
    assert s1.weight_versions == ["w1"]
    assert np.array_equal(s1.rollout_routed_experts, _expected_r3(0, 4))
    assert s1.prefix_cache_info.to_dict() == {"cached_tokens": 0, "total_prompt_tokens": 3}

    assert s2.tokens == _ACCUMULATED
    assert s2.response == tokenizer.decode([30, 31])
    assert s2.response_length == 2
    assert s2.loss_mask == [1, 1]
    assert s2.rollout_log_probs == [-0.5, -1.0]
    assert s2.status == Sample.Status.COMPLETED
    assert s2.weight_versions == ["w2"]
    assert np.array_equal(s2.rollout_routed_experts, _expected_r3(100, 8))
    assert s2.prefix_cache_info.to_dict() == {"cached_tokens": 5, "total_prompt_tokens": 7}

    # Overlay: template fields are the driver's, untouched by the wire.
    for s in samples:
        assert s.prompt == [{"role": "user", "content": "hi"}]
        assert s.label == "lbl"
        assert s.reward == 2.5
        assert s.routing_key == "routing-sid"
        assert s.train_metadata == {"loss": "ppo"}
        assert s.metadata["task"] == "t1"
    # Metadata application order: agent overrides the input's shared_key on every
    # sample; session_metadata (applied last, last sample only) overrides the
    # agent's max_trim_tokens plant.
    assert samples[0].metadata["shared_key"] == "from-agent"
    assert samples[0].metadata["max_trim_tokens"] == "agent-plant"
    assert samples[-1].metadata["max_trim_tokens"] == reply.session_metadata["max_trim_tokens"]
    assert samples[-1].metadata["accumulated_token_ids"] == _ACCUMULATED


async def test_assembled_samples_golden_merged(core):
    """multi_samples=False: turns merge into one trajectory Sample; the env
    tokens between turns get zero loss/logprob; the last turn's R3 is kept."""
    sid = await _make_session(core, _two_turn_records(), _ACCUMULATED)
    status, payload = await _collect_via_op(core, sid)
    assert status == 200
    samples, reply = _new_pipeline(payload, _input_sample(), multi_samples=False)
    (m,) = samples
    tokenizer = core.registry.tokenizer

    assert m.tokens == _ACCUMULATED
    assert m.response == tokenizer.decode([10, 11]) + tokenizer.decode([20, 21]) + tokenizer.decode([30, 31])
    assert m.response_length == 6
    assert m.loss_mask == [1, 1, 0, 0, 1, 1]
    assert m.rollout_log_probs == [-0.125, -0.25, 0.0, 0.0, -0.5, -1.0]
    assert m.status == Sample.Status.COMPLETED
    assert m.weight_versions == ["w1", "w2"]
    assert np.array_equal(m.rollout_routed_experts, _expected_r3(100, 8))
    assert m.prefix_cache_info.to_dict() == {"cached_tokens": 5, "total_prompt_tokens": 10}
    assert m.metadata["shared_key"] == "from-agent"
    assert m.metadata["accumulated_token_ids"] == _ACCUMULATED


@pytest.mark.parametrize("multi_samples", [False, True], ids=["merge", "multi"])
async def test_truncation_golden(core, multi_samples):
    """max_seq_len=8 strips one output token off the second turn (a turn-level
    budget applied before merge): the final sample ends TRUNCATED at 8 tokens
    with its per-token fields (including R3) trimmed in lockstep."""
    sid = await _make_session(core, _two_turn_records(), _ACCUMULATED)
    status, payload = await _collect_via_op(core, sid, multi_samples=multi_samples, max_seq_len=8)
    assert status == 200
    samples, _ = _new_pipeline(payload, _input_sample(), multi_samples=multi_samples)

    last = samples[-1]
    assert last.status == Sample.Status.TRUNCATED
    assert last.tokens == _ACCUMULATED[:8]
    if multi_samples:
        assert len(samples) == 2
        assert last.loss_mask == [1]
        assert last.rollout_log_probs == [-0.5]
    else:
        assert last.loss_mask == [1, 1, 0, 0, 1]
        assert last.rollout_log_probs == [-0.125, -0.25, 0.0, 0.0, -0.5]
    assert np.array_equal(last.rollout_routed_experts, _expected_r3(100, 8)[:-1])


async def test_session_metadata_matches_get_session(core):
    """The samples reply and the records GET must expose the same metadata dict
    (both are built by the extracted _session_metadata helper)."""
    sid = await _make_session(core, _two_turn_records(), _ACCUMULATED)
    _, payload = await _collect_via_op(core, sid)
    reply = decode_samples_reply(payload, Sample())

    response = await core.get_session(sid)
    assert response.status_code == 200
    assert reply.session_metadata == json.loads(response.body)["metadata"]
    assert reply.session_metadata["accumulated_token_ids"] == _ACCUMULATED


# ── empty_reason discriminator ──


async def test_no_records_reply(core):
    sid = await _make_session(core, [], None)
    status, payload = await _collect_via_op(core, sid)
    assert status == 200
    reply = decode_samples_reply(payload, Sample())
    assert reply.samples == [] and reply.empty_reason == "no_records"


async def test_all_truncated_reply(core):
    # max_seq_len=2 < the first turn's prompt+1: truncate_samples_by_total_tokens
    # drops every turn -> empty samples with the all_truncated reason; the old
    # pipeline returns [] on the same fixture (today's ABORTED path).
    records = _two_turn_records()
    sid = await _make_session(core, records, _ACCUMULATED)
    status, payload = await _collect_via_op(core, sid, max_seq_len=2)
    assert status == 200
    reply = decode_samples_reply(payload, Sample())
    assert reply.samples == [] and reply.empty_reason == "all_truncated"


# ── the 422 lane ──


async def test_broken_chain_returns_422_and_server_survives(core):
    # The accumulated sequence carries one token the records never produced ->
    # the cursor consistency assert fires -> 422 with the assertion text, and
    # the server keeps serving (the failure never escapes as an unhandled 500).
    sid = await _make_session(core, _two_turn_records(), _ACCUMULATED + [99])
    status, payload = await _collect_via_op(core, sid)
    assert status == 422
    assert "cursor" in payload.decode()

    health = await core.health()
    assert health.status_code == 200


# ── the HTTP surface: route order and error mapping through the real app ──


@pytest.fixture(scope="module")
def app_client():
    app = FastAPI()
    setup_session_routes(app, _UnusedBackend(), _ARGS)
    with TestClient(app) as client:
        yield client


def test_missing_session_returns_404(app_client):
    response = app_client.post(
        f"/sessions/{uuid.uuid4().hex}/samples", content=b'{"multi_samples":false,"max_seq_len":null}'
    )
    assert response.status_code == 404
    assert "not found" in response.json()["error"]


def test_samples_route_registered_before_catch_all_proxy(app_client):
    # The catch-all session_proxy would forward the request to the inference
    # backend (_UnusedBackend raises); the samples route must win instead and
    # answer with a decodable empty reply for a fresh session.
    sid = app_client.post("/sessions").json()["session_id"]
    response = app_client.post(f"/sessions/{sid}/samples", content=b'{"multi_samples":false,"max_seq_len":null}')
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    reply = decode_samples_reply(response.content, Sample())
    assert reply.empty_reason == "no_records", "catch-all session_proxy swallowed the samples route"
