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
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.fast.rollout.session.test_samples import _make_record

from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_reply
from miles.rollout.session.sessions import setup_session_routes
from miles.rollout.session.v2.core import SessionCore
from miles.rollout.session.v2.session_state import SessionRegistry
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.misc import function_registry
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

NUM_LAYERS = 3
TOPK = 2

_ARGS = SimpleNamespace(
    use_session_server="v2",
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
_AGENT_METADATA = {"shared_key": "from-agent", "agent_only": 1, "accumulated_token_ids": "agent-plant"}


async def _make_session(core, records, accumulated) -> str:
    response = await core.create_session()
    sid = json.loads(response.body)["session_id"]
    state = core.registry.sessions[sid]
    for i, record in enumerate(records):
        last = i == len(records) - 1
        state.active_leaf = state.tree.create_node(
            state.active_leaf,
            delta_messages=[],
            token_ids=list(accumulated) if (last and accumulated is not None) else [],
            completion_span=(0, 0),
            committed_at=record.timestamp,
            response_id="",
            record=record,
            finish_reason="",
        )
    return sid


async def _collect_via_op(core, sid, *, max_seq_len=None, agent_metadata=None):
    response = await core.collect_samples(sid, max_seq_len=max_seq_len, agent_metadata=agent_metadata)
    return response.status_code, response.body


def _new_pipeline(payload, input_sample):
    """What the driver does with the reply: decode only — per-sample metadata
    and rewards arrive already applied by the server-side merge."""
    reply = decode_samples_reply(payload, input_sample, fields=COMPUTED_FIELDS_V2)
    return reply.samples, reply


# ── golden assembly: exact expected Samples for the two-turn fixture ──


def _expected_r3(seed: int, num_tokens: int):
    return np.arange(seed, seed + num_tokens * NUM_LAYERS * TOPK, dtype=np.int32).reshape(num_tokens, NUM_LAYERS, TOPK)


async def test_assembled_samples_golden_merged(core):
    """Turns merge into one trajectory Sample; the env tokens between turns
    get zero loss/logprob; the last turn's R3 is kept."""
    sid = await _make_session(core, _two_turn_records(), _ACCUMULATED)
    status, payload = await _collect_via_op(core, sid, agent_metadata=_AGENT_METADATA)
    assert status == 200
    samples, reply = _new_pipeline(payload, _input_sample())
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
    # Overlay: template fields are the driver's, untouched by the wire.
    assert m.prompt == [{"role": "user", "content": "hi"}]
    assert m.label == "lbl"
    assert m.reward == 2.5
    assert m.routing_key == "routing-sid"
    assert m.train_metadata == {"loss": "ppo"}
    assert m.metadata["task"] == "t1"
    # Metadata layering (server-side merge, wire-carried): the agent's
    # semantic layer overrides the input's shared_key; the server's flat keys
    # land last, so the agent's accumulated_token_ids plant cannot survive.
    assert m.metadata["shared_key"] == "from-agent"
    assert m.metadata["agent_only"] == 1
    assert m.metadata["accumulated_token_ids"] == _ACCUMULATED
    assert reply.session_metadata["agent"] == _AGENT_METADATA


async def test_truncation_golden(core):
    """max_seq_len=8 strips one output token off the second turn (a turn-level
    budget applied before merge): the merged sample ends TRUNCATED at 8 tokens
    with its per-token fields (including R3) trimmed in lockstep."""
    sid = await _make_session(core, _two_turn_records(), _ACCUMULATED)
    status, payload = await _collect_via_op(core, sid, max_seq_len=8)
    assert status == 200
    samples, _ = _new_pipeline(payload, _input_sample())

    (last,) = samples
    assert last.status == Sample.Status.TRUNCATED
    assert last.tokens == _ACCUMULATED[:8]
    assert last.loss_mask == [1, 1, 0, 0, 1]
    assert last.rollout_log_probs == [-0.125, -0.25, 0.0, 0.0, -0.5]
    assert np.array_equal(last.rollout_routed_experts, _expected_r3(100, 8)[:-1])


async def test_session_metadata_matches_get_session(core):
    """The samples reply and the records GET must expose the same metadata dict
    (both are built by the extracted _session_metadata helper)."""
    sid = await _make_session(core, _two_turn_records(), _ACCUMULATED)
    _, payload = await _collect_via_op(core, sid)
    reply = decode_samples_reply(payload, Sample(), fields=COMPUTED_FIELDS_V2)

    response = await core.get_session(sid)
    assert response.status_code == 200
    assert reply.session_metadata == json.loads(response.body)["metadata"]
    assert reply.session_metadata["accumulated_token_ids"] == _ACCUMULATED


# ── empty_reason discriminator ──


async def test_no_records_reply(core):
    sid = await _make_session(core, [], None)
    status, payload = await _collect_via_op(core, sid)
    assert status == 200
    reply = decode_samples_reply(payload, Sample(), fields=COMPUTED_FIELDS_V2)
    assert reply.samples == [] and reply.empty_reason == "no_records"


async def test_all_truncated_reply(core):
    # max_seq_len=2 < the first turn's prompt+1: truncate_samples_by_total_tokens
    # drops every turn -> empty samples with the all_truncated reason; the old
    # pipeline returns [] on the same fixture (today's ABORTED path).
    records = _two_turn_records()
    sid = await _make_session(core, records, _ACCUMULATED)
    status, payload = await _collect_via_op(core, sid, max_seq_len=2)
    assert status == 200
    reply = decode_samples_reply(payload, Sample(), fields=COMPUTED_FIELDS_V2)
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


# ── the tree data plane: branches, trims, exactly-once, rewards ──


def _single_turn_record(prompt_ids, output_ids, weight_version="w1"):
    return _make_record(
        prompt_token_ids=list(prompt_ids),
        output_token_ids=list(output_ids),
        output_log_probs=[-0.1] * len(output_ids),
        cached_tokens=0,
        prompt_tokens=len(prompt_ids),
        weight_version=weight_version,
        routed_experts=_r3_b64(len(prompt_ids) + len(output_ids) - 1, seed=0),
    )


def _fabricate_node(state, parent, record, token_ids, *, completion_span, response_id=""):
    return state.tree.create_node(
        parent,
        delta_messages=[],
        token_ids=list(token_ids),
        completion_span=completion_span,
        committed_at=float(len(state.tree.nodes)),
        response_id=response_id,
        record=record,
        finish_reason="stop",
    )


async def _fresh_state(core):
    response = await core.create_session()
    sid = json.loads(response.body)["session_id"]
    return sid, core.registry.sessions[sid]


async def test_superseded_retry_leaf_is_trimmed(core):
    """A childless leaf with a later sibling is retry noise: one sample out."""
    sid, state = await _fresh_state(core)
    root = _fabricate_node(
        state, None, _single_turn_record([1, 2, 3], [10, 11]), [1, 2, 3, 10, 11], completion_span=(3, 5)
    )
    _fabricate_node(  # abandoned attempt
        state,
        root,
        _single_turn_record([1, 2, 3, 10, 11, 20], [30]),
        [1, 2, 3, 10, 11, 20, 30],
        completion_span=(6, 7),
    )
    retry = _fabricate_node(  # the retry that superseded it
        state,
        root,
        _single_turn_record([1, 2, 3, 10, 11, 21], [31]),
        [1, 2, 3, 10, 11, 21, 31],
        completion_span=(6, 7),
    )
    state.active_leaf = retry

    status, payload = await _collect_via_op(core, sid)
    assert status == 200
    reply = decode_samples_reply(payload, Sample(), fields=COMPUTED_FIELDS_V2)
    (m,) = reply.samples
    assert m.tokens == [1, 2, 3, 10, 11, 21, 31]


async def test_deep_abandoned_branch_survives_and_masks_shared_prefix(core):
    """A deep abandoned branch is data (subagent shape): two samples, and the
    shared root completion trains exactly once (earliest leaf owns it)."""
    sid, state = await _fresh_state(core)
    root = _fabricate_node(
        state, None, _single_turn_record([1, 2, 3], [10, 11]), [1, 2, 3, 10, 11], completion_span=(3, 5)
    )
    early_mid = _fabricate_node(
        state,
        root,
        _single_turn_record([1, 2, 3, 10, 11, 20], [30]),
        [1, 2, 3, 10, 11, 20, 30],
        completion_span=(6, 7),
    )
    early_leaf = _fabricate_node(
        state,
        early_mid,
        _single_turn_record([1, 2, 3, 10, 11, 20, 30, 40], [50]),
        [1, 2, 3, 10, 11, 20, 30, 40, 50],
        completion_span=(8, 9),
        response_id="early",
    )
    late_leaf = _fabricate_node(
        state,
        root,
        _single_turn_record([1, 2, 3, 10, 11, 21], [31]),
        [1, 2, 3, 10, 11, 21, 31],
        completion_span=(6, 7),
        response_id="late",
    )
    state.active_leaf = late_leaf

    status, payload = await _collect_via_op(core, sid, agent_metadata={"rewards": {"early": 0.25, "late": 1.0}})
    assert status == 200
    reply = decode_samples_reply(payload, Sample(), fields=COMPUTED_FIELDS_V2)
    early_sample, late_sample = reply.samples
    assert early_sample.tokens == early_leaf.token_ids
    assert late_sample.tokens == late_leaf.token_ids
    # The early leaf owns the shared root completion [3,5); the late leaf masks it.
    assert early_sample.loss_mask[:2] == [1, 1]
    assert late_sample.loss_mask[:2] == [0, 0]
    assert late_sample.loss_mask[-1] == 1  # its own completion still trains
    # Rewards keyed by response_id through the semantic layer.
    assert early_sample.reward == 0.25
    assert late_sample.reward == 1.0


async def test_picker_guardrail_rejects_longer_superseded_leaf(core):
    """A superseded leaf LONGER than every later sibling is not retry-shaped:
    fail loud (422) instead of silently guessing."""
    sid, state = await _fresh_state(core)
    root = _fabricate_node(
        state, None, _single_turn_record([1, 2, 3], [10, 11]), [1, 2, 3, 10, 11], completion_span=(3, 5)
    )
    _fabricate_node(  # abandoned but LONGER than the retry
        state,
        root,
        _single_turn_record([1, 2, 3, 10, 11, 20], [30, 31, 32]),
        [1, 2, 3, 10, 11, 20, 30, 31, 32],
        completion_span=(6, 9),
    )
    retry = _fabricate_node(
        state,
        root,
        _single_turn_record([1, 2, 3, 10, 11, 21], [31]),
        [1, 2, 3, 10, 11, 21, 31],
        completion_span=(6, 7),
    )
    state.active_leaf = retry

    status, payload = await _collect_via_op(core, sid)
    assert status == 422
    assert "not retry-shaped" in payload.decode()

    health = await core.health()
    assert health.status_code == 200


async def test_two_roots_yield_two_samples(core):
    """Zero-overlap branches (subagent forest) each assemble independently."""
    sid, state = await _fresh_state(core)
    main = _fabricate_node(
        state, None, _single_turn_record([1, 2, 3], [10, 11]), [1, 2, 3, 10, 11], completion_span=(3, 5)
    )
    sub = _fabricate_node(state, None, _single_turn_record([7, 8], [70, 71]), [7, 8, 70, 71], completion_span=(2, 4))
    state.active_leaf = sub

    status, payload = await _collect_via_op(core, sid)
    assert status == 200
    reply = decode_samples_reply(payload, Sample(), fields=COMPUTED_FIELDS_V2)
    first, second = reply.samples
    assert first.tokens == main.token_ids
    assert second.tokens == sub.token_ids
    tree = reply.session_metadata["tree"]
    assert [n["parent"] for n in tree["nodes"]] == [None, None]
    assert [leaf["node_id"] for leaf in tree["leaves"]] == [main.seq, sub.seq]


# ── the hook layer: custom pick/merge, contract enforcement ──


def _keep_all_picker(leaf_samples, session_metadata):
    return list(leaf_samples)


def _exploding_picker(leaf_samples, session_metadata):
    raise RuntimeError("policy bug")


def _impure_picker(leaf_samples, session_metadata):
    return [deepcopy(leaf_samples[0])]


async def _exploding_async_picker(leaf_samples, session_metadata):
    return leaf_samples


def _build_core_with_hooks(**hook_args) -> SessionCore:
    args = SimpleNamespace(**{**vars(_ARGS), **hook_args})
    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True)
    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=args.tito_model,
        chat_template_kwargs=args.apply_chat_template_kwargs,
        allowed_append_roles=args.tito_allowed_append_roles,
    )
    registry = SessionRegistry(args, tokenizer, tito_tokenizer=tito_tokenizer)
    return SessionCore(_UnusedBackend(), registry, args, args.session_server_instance_id)


async def _retry_shaped_session(core):
    sid, state = await _fresh_state(core)
    root = _fabricate_node(
        state, None, _single_turn_record([1, 2, 3], [10, 11]), [1, 2, 3, 10, 11], completion_span=(3, 5)
    )
    _fabricate_node(
        state,
        root,
        _single_turn_record([1, 2, 3, 10, 11, 20], [30]),
        [1, 2, 3, 10, 11, 20, 30],
        completion_span=(6, 7),
    )
    retry = _fabricate_node(
        state,
        root,
        _single_turn_record([1, 2, 3, 10, 11, 21], [31]),
        [1, 2, 3, 10, 11, 21, 31],
        completion_span=(6, 7),
    )
    state.active_leaf = retry
    return sid


async def test_custom_picker_keeps_abandoned_leaf(core):
    """A tree-RL style picker keeps everything: the abandoned retry leaf
    becomes a second sample instead of being trimmed."""
    with function_registry.temporary("test_hooks.keep_all", _keep_all_picker):
        hooked = _build_core_with_hooks(session_sample_picker_path="test_hooks.keep_all")
        sid = await _retry_shaped_session(hooked)
        response = await hooked.collect_samples(sid, max_seq_len=None)
        assert response.status_code == 200
        reply = decode_samples_reply(bytes(response.body), Sample(), fields=COMPUTED_FIELDS_V2)
        abandoned, retry = reply.samples
        assert abandoned.tokens == [1, 2, 3, 10, 11, 20, 30]
        assert retry.tokens == [1, 2, 3, 10, 11, 21, 31]
        # Exactly-once over the SURVIVING set: the abandoned (earlier) leaf now
        # owns the shared root completion; the retry masks it.
        assert abandoned.loss_mask[:2] == [1, 1]
        assert retry.loss_mask[:2] == [0, 0]


async def test_hook_exception_maps_to_422_with_identity(core):
    with function_registry.temporary("test_hooks.exploding", _exploding_picker):
        hooked = _build_core_with_hooks(session_sample_picker_path="test_hooks.exploding")
        sid = await _retry_shaped_session(hooked)
        response = await hooked.collect_samples(sid, max_seq_len=None)
        assert response.status_code == 422
        body = bytes(response.body).decode()
        assert "test_hooks.exploding" in body and "policy bug" in body


async def test_impure_picker_maps_to_422(core):
    with function_registry.temporary("test_hooks.impure", _impure_picker):
        hooked = _build_core_with_hooks(session_sample_picker_path="test_hooks.impure")
        sid = await _retry_shaped_session(hooked)
        response = await hooked.collect_samples(sid, max_seq_len=None)
        assert response.status_code == 422
        assert "subset" in bytes(response.body).decode()


def test_async_hook_rejected_at_load():
    with function_registry.temporary("test_hooks.async_picker", _exploding_async_picker):
        with pytest.raises(ValueError, match="async"):
            _build_core_with_hooks(session_sample_picker_path="test_hooks.async_picker")


# ── the HTTP surface: route order and error mapping through the real app ──


@pytest.fixture(scope="module")
def app_client():
    app = FastAPI()
    setup_session_routes(app, _UnusedBackend(), _ARGS)
    with TestClient(app) as client:
        yield client


def test_missing_session_returns_404(app_client):
    response = app_client.post(f"/sessions/{uuid.uuid4().hex}/samples", content=b'{"max_seq_len":null}')
    assert response.status_code == 404
    assert "not found" in response.json()["error"]


def test_samples_route_registered_before_catch_all_proxy(app_client):
    # The catch-all session_proxy would forward the request to the inference
    # backend (_UnusedBackend raises); the samples route must win instead and
    # answer with a decodable empty reply for a fresh session.
    sid = app_client.post("/sessions").json()["session_id"]
    response = app_client.post(f"/sessions/{sid}/samples", content=b'{"max_seq_len":null}')
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    reply = decode_samples_reply(response.content, Sample(), fields=COMPUTED_FIELDS_V2)
    assert reply.empty_reason == "no_records", "catch-all session_proxy swallowed the samples route"
