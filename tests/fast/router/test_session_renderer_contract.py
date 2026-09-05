import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from tests.fast.router.test_linear_trajectory import _MockTITOTokenizer

import miles.rollout.session.core as core_v1
import miles.rollout.session.v2.core as core_v2
from miles.rollout.session.errors import MessageValidationError, UpstreamResponseError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.sessions import setup_session_routes
from miles.rollout.session.v2.session_state import SessionRegistryV2
from miles.utils.chat_template_utils import TITOTokenizerType, resolve_fixed_chat_template


def _response():
    return {
        "status_code": 200,
        "headers": {"content-type": "application/json"},
        "response_body": json.dumps(
            {
                "id": "completion",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                        "meta_info": {"output_token_logprobs": [[-0.1, 1]], "completion_tokens": 1},
                    }
                ],
            }
        ).encode(),
    }


class _Backend:
    def __init__(self):
        self.result = _response()
        self.calls = []
        self.blocked = False
        self.arrivals = asyncio.Queue()

    async def do_proxy(self, request, path, *, body, headers):
        self.calls.append(json.loads(body))
        if self.blocked:
            result = asyncio.get_running_loop().create_future()
            self.arrivals.put_nowait(result)
            return await result
        return deepcopy(self.result)

    async def next_arrival(self):
        return await asyncio.wait_for(self.arrivals.get(), timeout=5)


def _args(version, **overrides):
    return SimpleNamespace(
        **{
            "use_session_server": version,
            "hf_checkpoint": "Qwen/Qwen3-0.6B",
            "chat_template_path": None,
            "apply_chat_template_kwargs": {"enable_thinking": False},
            "tito_model": "default",
            "session_sample_picker_path": "miles.rollout.session.v2.picker_hub.drop_retries",
            "session_sample_postprocessor_path": "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
            **overrides,
        }
    )


@pytest.fixture(params=["v1", "v2"])
def core(request):
    args = _args(request.param)
    tokenizer = _MockTITOTokenizer(None, chat_template_kwargs={"add_vision_id": False})
    registry_cls = SessionRegistryV2 if request.param == "v2" else SessionRegistry
    core_cls = core_v2.SessionCoreV2 if request.param == "v2" else core_v1.SessionCore
    registry = registry_cls(args, None, tito_tokenizer=tokenizer)
    return core_cls(_Backend(), registry, args)


async def _chat(core, sid, **values):
    body = {"messages": [{"role": "user", "content": "hi"}], **values}
    return await core.chat_completions(sid, method="POST", query="", headers={}, body=json.dumps(body).encode())


def _records(core, state):
    if isinstance(core, core_v2.SessionCoreV2):
        return [node.record for node in state.tree.nodes]
    return state.records


async def test_successful_commit_pins_and_sampling_fields_remain_per_turn(core):
    sid = core.registry.create_session()
    state = core.registry.get_session(sid)
    core.backend.blocked = True
    task = asyncio.create_task(_chat(core, sid))
    pending = await core.backend.next_arrival()
    assert state.session_tito_tokenizer is None
    pending.set_result(_response())
    assert (await task).status_code == 200
    pinned = state.session_tito_tokenizer
    assert pinned is not core.registry.tito_tokenizer
    assert pinned.chat_template_kwargs == {"add_vision_id": False}

    core.backend.blocked = False
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "continue"},
    ]
    assert (await _chat(core, sid, messages=messages, temperature=0.5, max_tokens=12)).status_code == 200
    assert core.backend.calls[-1]["temperature"] == 0.5
    assert core.backend.calls[-1]["max_tokens"] == 12
    assert state.session_tito_tokenizer is pinned
    before = list(_records(core, state))
    with pytest.raises(MessageValidationError, match="chat template configuration cannot change"):
        await _chat(core, sid, chat_template_kwargs={"add_vision_id": True})
    assert len(core.backend.calls) == 2
    assert _records(core, state) == before


@pytest.mark.parametrize("failure", ["backend", "extract", "json", "render", "commit", "record"])
async def test_failed_first_turn_does_not_pin_and_different_retry_can_commit(core, monkeypatch, failure):
    sid = core.registry.create_session()
    state = core.registry.get_session(sid)
    module = core_v2 if isinstance(core, core_v2.SessionCoreV2) else core_v1

    def fail(*args, **kwargs):
        raise ValueError("injected failure")

    with monkeypatch.context() as patch:
        if failure == "backend":
            core.backend.result["status_code"] = 400
        elif failure == "extract":
            core.backend.result["response_body"] = b'{"choices": [{}]}'
        elif failure == "json":
            core.backend.result["response_body"] = b"not json"
        elif failure == "render":
            patch.setattr(_MockTITOTokenizer, "apply_chat_template", fail)
        elif failure == "record":
            patch.setattr(module, "SessionRecord", fail)
        elif isinstance(core, core_v2.SessionCoreV2):
            patch.setattr(module, "commit_generation", fail)
        else:
            patch.setattr(state, "update_pretokenized_state", fail)

        if failure == "backend":
            assert (await _chat(core, sid)).status_code == 400
        else:
            error = UpstreamResponseError if failure == "extract" else ValueError
            with pytest.raises(error):
                await _chat(core, sid)

    assert state.session_tito_tokenizer is None
    assert not _records(core, state)
    core.backend.result = _response()
    assert (await _chat(core, sid, chat_template_kwargs={"add_vision_id": True})).status_code == 200
    assert state.session_tito_tokenizer.chat_template_kwargs == {"add_vision_id": True}


async def test_close_during_first_backend_call_does_not_pin(core):
    sid = core.registry.create_session()
    state = core.registry.get_session(sid)
    core.backend.blocked = True
    task = asyncio.create_task(_chat(core, sid))
    pending = await core.backend.next_arrival()
    await core.delete_session(sid)
    pending.set_result(_response())
    assert (await task).status_code == 200
    assert state.session_tito_tokenizer is None
    assert not _records(core, state)


@pytest.mark.parametrize("changed", [1, "true", None, False])
async def test_full_nested_identity_preserves_types_and_record_mutation_cannot_change_pin(core, changed):
    sid = core.registry.create_session()
    kwargs = {"options": {"flags": [True]}, "add_vision_id": False}
    assert (await _chat(core, sid, chat_template_kwargs=kwargs)).status_code == 200
    state = core.registry.get_session(sid)
    pinned = state.session_tito_tokenizer
    _records(core, state)[0].request["chat_template_kwargs"]["options"]["flags"][0] = changed
    assert pinned.chat_template_kwargs == kwargs
    with pytest.raises(MessageValidationError, match="chat template configuration cannot change"):
        await _chat(core, sid, chat_template_kwargs={"options": {"flags": [changed]}})
    assert len(core.backend.calls) == 1


async def test_mapping_order_is_not_a_renderer_change(core):
    sid = core.registry.create_session()
    first = {"options": {"a": True, "b": [1, 2]}, "add_vision_id": False}
    assert (await _chat(core, sid, chat_template_kwargs=first)).status_code == 200
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "continue"},
    ]
    reordered = {"add_vision_id": False, "options": {"b": [1, 2], "a": True}}
    assert (await _chat(core, sid, messages=messages, chat_template_kwargs=reordered)).status_code == 200


async def test_omitted_override_is_resolved_from_startup_and_cannot_change_pin(core):
    sid = core.registry.create_session()
    assert (await _chat(core, sid, chat_template_kwargs={"add_vision_id": True})).status_code == 200
    with pytest.raises(MessageValidationError, match="chat template configuration cannot change"):
        await _chat(core, sid)
    assert len(core.backend.calls) == 1


@pytest.mark.parametrize("same_renderer", [False, True])
@pytest.mark.parametrize("first_finisher", [0, 1])
async def test_first_commit_wins_renderer_race_and_same_renderer_v2_keeps_siblings(
    core, same_renderer, first_finisher
):
    sid = core.registry.create_session()
    state = core.registry.get_session(sid)
    core.backend.blocked = True
    configs = [False, False if same_renderer else True]
    tasks, pending = [], []
    for config in configs:
        tasks.append(asyncio.create_task(_chat(core, sid, chat_template_kwargs={"add_vision_id": config})))
        pending.append(await core.backend.next_arrival())
    assert state.session_tito_tokenizer is None

    pending[first_finisher].set_result(_response())
    assert (await tasks[first_finisher]).status_code == 200
    pinned = state.session_tito_tokenizer
    assert pinned.chat_template_kwargs == {"add_vision_id": configs[first_finisher]}
    pending[1 - first_finisher].set_result(_response())
    assert (await tasks[1 - first_finisher]).status_code == 200
    expected = 2 if same_renderer and isinstance(core, core_v2.SessionCoreV2) else 1
    assert len(_records(core, state)) == expected
    assert all(
        record.request["chat_template_kwargs"] == pinned.chat_template_kwargs for record in _records(core, state)
    )
    assert state.session_tito_tokenizer is pinned

    with pytest.raises(MessageValidationError, match="chat template configuration cannot change"):
        await _chat(core, sid, chat_template_kwargs={"add_vision_id": not configs[first_finisher]})
    assert len(core.backend.calls) == 2


@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_http_conflict_uses_existing_error_envelope_before_backend(version):
    app, backend = FastAPI(), _Backend()
    setup_session_routes(app, backend, _args(version))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        sid = (await client.post("/sessions")).json()["session_id"]
        url = f"/sessions/{sid}/v1/chat/completions"
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert (await client.post(url, json=body)).status_code == 200
        response = await client.post(url, json={**body, "chat_template_kwargs": {"enable_thinking": True}})
        assert response.status_code == 400
        assert response.json() == {
            "error": "chat template configuration cannot change within a session: "
            "established={'enable_thinking': False}, requested={'enable_thinking': True}"
        }
        assert len(backend.calls) == 1


@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_qwen_http_default_and_selected_reasoning_match_outbound_and_record(version):
    path, kwargs = resolve_fixed_chat_template(TITOTokenizerType.QWEN38_SMALL)
    app, backend = FastAPI(), _Backend()
    setup_session_routes(
        app,
        backend,
        _args(
            version,
            tito_model=TITOTokenizerType.QWEN38_SMALL.value,
            chat_template_path=path,
            apply_chat_template_kwargs=kwargs,
        ),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for request_kwargs, expected in [
            ({}, {"preserve_thinking": True, "reasoning_effort": "xhigh"}),
            (
                {"reasoning_effort": "low"},
                {"preserve_thinking": True, "reasoning_effort": "low", "enable_thinking": True},
            ),
        ]:
            sid = (await client.post("/sessions")).json()["session_id"]
            response = await client.post(
                f"/sessions/{sid}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], **request_kwargs},
            )
            assert response.status_code == 200
            assert backend.calls[-1]["chat_template_kwargs"] == expected
            session = (await client.get(f"/sessions/{sid}")).json()
            assert session["records"][0]["request"] == backend.calls[-1]
