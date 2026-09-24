"""Recorded sessions on the Tinker gateway (renderer, collector, routes, wiring) on CPU with the fake backend."""

import json
import time
import types

import httpx
import pytest
from tests.fast.tinker.harness import make_service

from miles.tinker.arguments import _configure_tito
from miles.tinker.core.prompt_renderer import PromptRenderer
from miles.tinker.core.tinker_session_server import (
    SamplingBackendError,
    SessionLimitError,
    SessionNotFoundError,
    TrajectoryCollector,
    TrajectorySession,
    TruncatedGenerationError,
    Turn,
    TurnRequest,
)
from miles.tinker.core.types import OwnershipError, UserInputError
from miles.tinker.server.app import build_app
from miles.tinker.server.session_routes import setup_session_routes
from miles.tinker.session_setup import build_session_app
from miles.utils.chat_template_utils import TEMPLATE_DIR, get_tito_tokenizer, resolve_fixed_chat_template, template
from miles.utils.chat_template_utils.message_matcher_hub import strict_message_matches
from miles.utils.processing_utils import load_tokenizer

MODEL = "Qwen/Qwen3-0.6B"
TENANT, OTHER = "tenant-a", "tenant-b"
SID = "session-" + "0" * 32
USER = [{"role": "user", "content": "Name a color."}]
TOOLS = [{"type": "function", "function": {"name": "ls", "parameters": {"type": "object", "properties": {}}}}]
TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "ls", "arguments": '{"path": "."}'}}
DROP = object()  # a chat field left out of the request body


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(MODEL)


@pytest.fixture(scope="module")
def tito_pair():
    template_path, kwargs = resolve_fixed_chat_template("qwen3")
    hf = load_tokenizer(MODEL, chat_template_path=template_path)
    return hf, get_tito_tokenizer(hf, "qwen3", chat_template_kwargs=kwargs)


def _plain_renderer(tokenizer) -> PromptRenderer:
    return PromptRenderer(tokenizer, get_tito_tokenizer(tokenizer, "default"), inherit=False)


def _tito_renderer(tito_pair) -> PromptRenderer:
    hf, tito = tito_pair
    return PromptRenderer(hf, tito, inherit=True, message_matcher=strict_message_matches)


def _record(session, renderer, messages, reply_ids, *, finish="stop", stop=None, budget=8192):
    """Render, then commit a reply the way the collector does; returns (rendered, assistant message)."""
    rendered = renderer.prepare_pretokenized(session, messages, None, None, max_new_tokens=8, budget=budget)
    turn = Turn(
        input_ids=rendered.prompt_token_ids,
        output_ids=list(reply_ids),
        logprobs=[0.0] * len(reply_ids),
        finish_reason=finish,
        inherits=rendered.inherits,
        reset_reason=rendered.reset_reason,
        parent=rendered.parent,
        request_args=rendered.request_args,
    )
    message, turn.ended_on_stop = renderer.assistant_message(turn, stop)
    turn.messages = [*messages, message]
    session.turns.append(turn)
    return rendered, message


def _reply(hf, text: str) -> list[int]:
    return hf.encode(text, add_special_tokens=False) + [hf.convert_tokens_to_ids("<|im_end|>")]


# --- renderer (miles/tinker/core/prompt_renderer.py) ---------------------------------------------


def test_renderer_first_turn_is_a_root_and_a_resend_is_a_retry(tito_pair):
    renderer, session = _tito_renderer(tito_pair), TrajectorySession(SID, TENANT)
    first, _ = _record(session, renderer, USER, _reply(tito_pair[0], "Blue."))
    assert (first.parent, first.inherits, first.reset_reason) == (None, False, "first")
    resend = renderer.prepare_pretokenized(session, USER, None, None, max_new_tokens=8, budget=8192)
    assert (resend.parent, resend.inherits, resend.reset_reason) == (None, False, "retry")
    edited = renderer.prepare_pretokenized(
        session, [{"role": "user", "content": "Name a fruit."}], None, None, max_new_tokens=8, budget=8192
    )
    assert (edited.parent, edited.reset_reason) == (None, "rewrite")


def test_renderer_tito_turn_extends_the_parent_prefix(tito_pair):
    renderer, session = _tito_renderer(tito_pair), TrajectorySession(SID, TENANT)
    _, reply = _record(session, renderer, USER, _reply(tito_pair[0], "Blue."))
    follow = [*USER, reply, {"role": "user", "content": "Another one."}]
    rendered = renderer.prepare_pretokenized(session, follow, None, None, max_new_tokens=8, budget=8192)
    parent = session.turns[0]
    prefix = [*parent.input_ids, *parent.output_ids]
    assert (rendered.parent, rendered.inherits, rendered.reset_reason) == (0, True, None)
    assert rendered.prompt_token_ids[: len(prefix)] == prefix
    budgeted = renderer.prepare_pretokenized(session, follow, None, None, max_new_tokens=8, budget=len(prefix) + 4)
    assert (budgeted.inherits, budgeted.reset_reason) == (False, "budget")


def test_renderer_reply_ended_on_a_stop_string_forces_a_full_render(tito_pair):
    hf = tito_pair[0]
    renderer, session = _tito_renderer(tito_pair), TrajectorySession(SID, TENANT)
    _, reply = _record(session, renderer, USER, hf.encode("1, 2, 3", add_special_tokens=False), stop=["3"])
    assert reply["content"] == "1, 2, " and session.turns[0].ended_on_stop
    follow = [*USER, reply, {"role": "user", "content": "Now say done."}]
    rendered = renderer.prepare_pretokenized(session, follow, None, None, max_new_tokens=8, budget=8192)
    assert (rendered.parent, rendered.inherits, rendered.reset_reason) == (0, False, "stop_string")
    assert "<|im_end|>\n<|im_start|>user\nNow say done." in hf.decode(rendered.prompt_token_ids)


def test_renderer_merge_failure_falls_back_to_the_full_render(tito_pair):
    renderer, session = _tito_renderer(tito_pair), TrajectorySession(SID, TENANT)
    _, reply = _record(session, renderer, USER, _reply(tito_pair[0], "Blue."))
    appended = [*USER, reply, {"role": "assistant", "content": "", "tool_calls": [{}]}]
    with pytest.raises(UserInputError):  # the merge raises TypeError; the full render refuses the same input (400)
        renderer.prepare_pretokenized(session, appended, None, None, max_new_tokens=8, budget=8192)


def test_renderer_full_render_goes_through_the_miles_renderer():
    hf = load_tokenizer(MODEL, chat_template_path=f"{TEMPLATE_DIR}/qwen3.5_fixed.jinja")
    messages = [
        *USER,
        {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
    ]
    rendered = _plain_renderer(hf).prepare_pretokenized(
        TrajectorySession(SID, TENANT), messages, TOOLS, None, max_new_tokens=8, budget=8192
    )
    expected = template.apply_chat_template(
        messages, tokenizer=hf, tools=TOOLS, add_generation_prompt=True, tokenize=True
    )
    assert rendered.prompt_token_ids == list(expected) and rendered.reset_reason == "no_tito"


@pytest.mark.parametrize(
    "override", [{"chat_template": "X"}, {"truncation": True, "max_length": 3}, {"tokenize": False}, "not an object"]
)
def test_renderer_chat_template_kwargs_cannot_set_render_arguments(tokenizer, override):
    with pytest.raises(UserInputError):
        _plain_renderer(tokenizer).prepare_pretokenized(
            TrajectorySession(SID, TENANT), USER, None, override, max_new_tokens=8, budget=8192
        )


def test_renderer_template_variables_still_render(tokenizer):
    rendered = _plain_renderer(tokenizer).prepare_pretokenized(
        TrajectorySession(SID, TENANT), USER, None, {"enable_thinking": False}, max_new_tokens=8, budget=8192
    )
    assert rendered.prompt_token_ids


@pytest.mark.parametrize(
    "messages",
    ["not a list", [], [{"content": "no role"}], [*USER, {"role": "assistant", "content": "", "tool_calls": ["x"]}]],
)
def test_renderer_malformed_messages_are_user_errors(tokenizer, messages):
    with pytest.raises(UserInputError):
        _plain_renderer(tokenizer).prepare_pretokenized(
            TrajectorySession(SID, TENANT), messages, None, None, max_new_tokens=8, budget=8192
        )


def test_renderer_assistant_message_strips_only_a_matching_stop_string(tokenizer):
    renderer = _plain_renderer(tokenizer)
    ids = tokenizer.encode("1, 2, 3", add_special_tokens=False)
    turn = Turn(input_ids=[1], output_ids=ids, logprobs=[0.0] * len(ids), finish_reason="stop")
    assert renderer.assistant_message(turn, ["3"]) == ({"role": "assistant", "content": "1, 2, "}, True)
    assert renderer.assistant_message(turn, ["9"]) == ({"role": "assistant", "content": "1, 2, 3"}, False)
    turn.finish_reason = "length"
    assert renderer.assistant_message(turn, ["3"]) == ({"role": "assistant", "content": "1, 2, 3"}, False)
    assert turn.ended_on_stop is False  # the renderer never writes to the Turn; the collector does


# --- collector (miles/tinker/core/tinker_session_server.py) --------------------------------------


@pytest.fixture
def gateway(tmp_path, tito_pair):
    """A TinkerService on the fake backend and a TITO collector over it."""
    service = make_service(tmp_path, vocab_size=160000, max_tokens_per_datum=8192)
    return service, TrajectoryCollector(service, _tito_renderer(tito_pair), session_ttl_s=3600.0)


def _sampling_session(service, tenant: str = TENANT) -> str:
    session_id = service.create_session(tenant)
    body = {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": "base"}
    return service.create_sampling_session(tenant, body)


def _turn(messages=USER, **sampling) -> TurnRequest:
    return TurnRequest(messages=messages, tools=None, sampling_params={"max_tokens": 4, **sampling})


def test_collector_bind_checks_the_sampling_session(gateway):
    service, collector = gateway
    with pytest.raises(UserInputError):
        collector.create_session(SID, TENANT)
    with pytest.raises(UserInputError):
        collector.create_session(SID, TENANT, sampling_session_id=["x"])
    with pytest.raises(OwnershipError):
        collector.create_session(SID, TENANT, sampling_session_id=_sampling_session(service, OTHER))
    with pytest.raises(UserInputError):
        collector.create_session(SID, TENANT, sampling_session_id="sampling-unknown")


@pytest.mark.parametrize("length,ok", [(1, False), (31, False), (32, True), (128, True), (129, False)])
def test_collector_session_ids_are_32_to_128_chars(gateway, length, ok):
    service, collector = gateway
    ssid = _sampling_session(service)
    if ok:
        collector.create_session("s" * length, TENANT, sampling_session_id=ssid)
    else:
        with pytest.raises(UserInputError):
            collector.create_session("s" * length, TENANT, sampling_session_id=ssid)


@pytest.mark.parametrize("client_cap,effective", [(None, 8192), (4096, 4096), (10**9, 8192)])
def test_collector_datum_budget_never_exceeds_the_gateway_cap(gateway, client_cap, effective):
    service, collector = gateway
    session = collector.create_session(
        SID, TENANT, sampling_session_id=_sampling_session(service), max_datum_tokens=client_cap
    )
    assert collector.datum_budget(session) == effective


async def test_collector_records_the_sample_and_names_the_base_model(gateway):
    service, collector = gateway
    collector.create_session(SID, TENANT, sampling_session_id=_sampling_session(service))
    result = await collector.complete(SID, _turn())
    assert list(result.turn.output_ids) == [1, 2] and len(result.turn.logprobs) == 2
    assert (result.turn.finish_reason, result.model) == ("stop", "base")


async def test_collector_session_and_turn_caps_answer_429(tmp_path, tito_pair):
    service = make_service(tmp_path, vocab_size=160000)
    collector = TrajectoryCollector(
        service, _tito_renderer(tito_pair), session_ttl_s=3600.0, max_sessions_per_tenant=1, max_turns_per_session=1
    )
    ssid = _sampling_session(service)
    collector.create_session(SID, TENANT, sampling_session_id=ssid)
    with pytest.raises(SessionLimitError) as too_many_sessions:
        collector.create_session("t" * 32, TENANT, sampling_session_id=ssid)
    await collector.complete(SID, _turn())
    with pytest.raises(SessionLimitError) as too_many_turns:
        await collector.complete(SID, _turn())
    assert too_many_sessions.value.status_code == too_many_turns.value.status_code == 429


async def test_collector_delete_during_the_render_samples_nothing(gateway):
    service, collector = gateway
    collector.create_session(SID, TENANT, sampling_session_id=_sampling_session(service))
    render = collector.renderer.prepare_pretokenized

    def render_then_delete(*args, **kwargs):
        rendered = render(*args, **kwargs)
        collector.delete_session(SID, TENANT)
        return rendered

    collector.renderer.prepare_pretokenized = render_then_delete
    with pytest.raises(SessionNotFoundError):
        await collector.complete(SID, _turn())
    assert service.backend.named("sample") == []


async def test_collector_engine_failure_is_a_502_and_records_nothing(gateway):
    service, collector = gateway
    session = collector.create_session(SID, TENANT, sampling_session_id=_sampling_session(service))
    service.backend.fail_on["sample"] = {"error": "engine down"}
    with pytest.raises(SamplingBackendError) as failure:
        await collector.complete(SID, _turn())
    assert failure.value.status_code == 502 and session.turns == []


async def test_collector_expired_lease_refuses_the_next_turn(gateway):
    service, collector = gateway
    collector.create_session(SID, TENANT, sampling_session_id=_sampling_session(service))
    for record in service.sessions.values():
        record.last_heartbeat -= 10**6
    await service._expire_sessions()
    with pytest.raises(UserInputError, match="or its lease expired"):
        await collector.complete(SID, _turn())


async def test_collector_cut_reply_flags_its_lineage_and_strict_mode_refuses(gateway):
    service, collector = gateway
    collector.create_session(SID, TENANT, sampling_session_id=_sampling_session(service))

    async def cut(payload, lora_name, lora_path=None):
        return {"sequences": [{"tokens": [1], "logprobs": [0.0], "stop_reason": "length"}]}

    service.backend.sample = cut
    first = await collector.complete(SID, _turn())
    follow = [*USER, first.assistant_message, {"role": "user", "content": "Go on."}]
    assert first.turn.finish_reason == "length"
    assert (await collector.complete(SID, _turn(follow))).turn.after_truncation
    collector.strict_truncation = True
    with pytest.raises(TruncatedGenerationError) as refused:
        await collector.complete(SID, _turn([*follow, {"role": "assistant", "content": "x"}, *USER]))
    assert refused.value.status_code == 409


def test_collector_sweeps_idle_sessions(gateway):
    service, collector = gateway
    collector.create_session(SID, TENANT, sampling_session_id=_sampling_session(service))
    assert collector.sweep(now=time.time() + 3601) == 1 and collector.sessions == {}


# --- routes (miles/tinker/server/session_routes.py, oai_shapes.py) -------------------------------


@pytest.fixture
async def client(gateway):
    service, collector = gateway
    app = build_app(service)
    setup_session_routes(app, collector, max_body_bytes=4096)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as http:
        http.ssid = _sampling_session(service)
        http.other_ssid = _sampling_session(service, OTHER)
        yield http


def _key(tenant: str = TENANT) -> dict:
    return {"X-API-Key": tenant}


def _chat_body(**extra) -> dict:
    return {"model": "base", "messages": USER, "max_tokens": 4, **extra}


async def test_routes_errors_keep_their_status_through_one_handler(client):
    unknown = await client.get(f"/oai/sessions/{SID}", headers=_key())
    assert unknown.status_code == 404 and "bind it first" in unknown.json()["error"]
    assert (await client.post(f"/oai/sessions/{SID}", json={}, headers=_key())).status_code == 400
    assert (await client.post(f"/oai/sessions/{SID}", json={"sampling_session_id": client.ssid})).status_code == 400
    foreign = await client.post(
        f"/oai/sessions/{SID}", json={"sampling_session_id": client.other_ssid}, headers=_key()
    )
    assert foreign.status_code == 403
    await client.post(f"/oai/sessions/{SID}", json={"sampling_session_id": client.ssid}, headers=_key())
    assert (await client.get(f"/oai/sessions/{SID}", headers=_key(OTHER))).status_code == 403
    assert (await client.get("/api/v1/samplers/sampling-unknown", headers=_key())).status_code == 400


@pytest.mark.parametrize(
    "content",
    [b'{"sampling_session_id": "\xff"}', b'{"a":' + b"[" * 100000 + b"]" * 100000 + b"}", b"[1, 2]", b"x" * 5000],
)
async def test_routes_bad_bodies_answer_400(client, content):
    response = await client.post(f"/oai/sessions/{SID}", content=content, headers=_key())
    assert response.status_code == 400


@pytest.mark.parametrize(
    "extra",
    [
        {"n": 2},
        {"stream": True},
        {"max_tokens": DROP},
        {"max_tokens": 0},
        {"temperature": None},
        {"temperature": "hot"},
        {"model": 123},
        {"stop": [1]},
        {"chat_template_kwargs": "x"},
    ],
)
async def test_routes_chat_request_validation(client, extra):
    await client.post(f"/oai/sessions/{SID}", json={"sampling_session_id": client.ssid}, headers=_key())
    body = {key: value for key, value in _chat_body(**extra).items() if value is not DROP}
    assert (await client.post(f"/oai/sessions/{SID}/v1/chat/completions", json=body)).status_code == 400


async def test_routes_chat_response_names_the_sampling_model(client):
    await client.post(f"/oai/sessions/{SID}", json={"sampling_session_id": client.ssid}, headers=_key())
    body = {"model": "gpt-4o", "messages": USER, "max_completion_tokens": 4}
    response = await client.post(f"/oai/sessions/{SID}/v1/chat/completions", json=body)
    payload = response.json()
    assert response.status_code == 200 and payload["model"] == "base"
    assert payload["choices"][0]["message"]["role"] == "assistant" and payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["total_tokens"] == payload["usage"]["prompt_tokens"] + 2


async def test_routes_bind_export_delete_round_trip(client):
    bound = await client.post(
        f"/oai/sessions/{SID}", json={"sampling_session_id": client.ssid, "max_datum_tokens": 10**9}, headers=_key()
    )
    assert bound.json()["max_datum_tokens"] == 8192
    await client.post(f"/oai/sessions/{SID}/v1/chat/completions", json=_chat_body())
    exported = (await client.get(f"/oai/sessions/{SID}", headers=_key())).json()
    assert [(t["parent"], t["inherits"], t["reset_reason"]) for t in exported["turns"]] == [(None, False, "first")]
    assert (await client.delete(f"/oai/sessions/{SID}", headers=_key())).json() == {"session_id": SID, "deleted": True}
    assert (await client.get(f"/oai/sessions/{SID}", headers=_key())).status_code == 404


# --- wiring (miles/tinker/session_setup.py, miles/tinker/arguments.py) ---------------------------


def _serve_args(**overrides):
    base = dict(
        hf_checkpoint=MODEL,
        chat_template_path=None,
        tinker_tito_model=None,
        tinker_session_server=True,
        apply_chat_template_kwargs=None,
        session_message_matcher="strict",
        tinker_session_ttl_s=3600.0,
        tinker_session_strict_truncation=False,
        tinker_session_max_body_bytes=16 * 1024 * 1024,
    )
    return types.SimpleNamespace(**{**base, **overrides})


@pytest.mark.parametrize(
    "selector,matcher",
    [
        ("strict", "strict_message_matches"),
        ("loose_tool_call", "loose_tool_call_message_matches"),
        ("role_content_only", "role_content_only_message_matches"),
    ],
)
def test_setup_matcher_follows_the_flag(tmp_path, selector, matcher):
    _, collector = build_session_app(make_service(tmp_path), args=_serve_args(session_message_matcher=selector))
    assert collector.renderer.message_matcher.__wrapped__.__name__ == matcher


def test_setup_invalid_matcher_fails_at_startup(tmp_path):
    with pytest.raises(ValueError, match="session-message-matcher"):
        build_session_app(make_service(tmp_path), args=_serve_args(session_message_matcher="typo"))


def test_setup_without_tito_every_turn_is_a_full_render(tmp_path):
    _, collector = build_session_app(make_service(tmp_path), args=_serve_args())
    assert collector.renderer.inherit is False and collector.renderer.max_trim_tokens == 0
    assert type(collector.renderer.tito_tokenizer).__name__ == "TITOTokenizer"  # the default family


def test_setup_tito_model_needs_the_session_server_and_installs_its_template():
    with pytest.raises(ValueError, match="requires --tinker-session-server"):
        _configure_tito(_serve_args(tinker_tito_model="qwen3", tinker_session_server=False))
    args = _serve_args(tinker_tito_model="qwen3")
    _configure_tito(args)
    assert args.chat_template_path.endswith("qwen3_fixed.jinja")
    assert json.dumps(args.apply_chat_template_kwargs) == json.dumps(resolve_fixed_chat_template("qwen3")[1])
