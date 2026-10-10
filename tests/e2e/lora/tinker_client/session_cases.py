import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers.role_colon import RoleColonRenderer

import tinker
from miles.tinker.client.rendering import ChatRequest, render_prompt
from miles.tinker.client.server import SessionServer
from miles.tinker.client.session import ChatSession


class Tokenizer:
    bos_token = None
    eos_token_id = None

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(" hello\n\nUser:" if token == 1000 else chr(token) for token in tokens)


def make_session():
    sequence = tinker.SampledSequence(
        stop_reason="stop", sequence_id="s", tokens_np=np.array([1000]), logprobs_np=np.array([-0.25])
    )
    client = SimpleNamespace(sample_async=AsyncMock(return_value=tinker.SampleResponse(sequences=[sequence])))
    return ChatSession(
        TinkerTokenCompleter(client, max_tokens=32, temperature=0.7),
        RoleColonRenderer(Tokenizer()),
        max_datum_tokens=1024,
    )


@pytest.mark.asyncio
async def test_renderer_owns_every_prompt_and_trace_keeps_raw_tokens():
    session = make_session()
    histories = [
        [{"role": "user", "content": "original"}],
        [
            {"role": "user", "content": "edited"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "continue"},
        ],
        [{"role": "user", "content": "summary"}],
        [{"role": "user", "content": "summary"}],
    ]
    for messages in histories:
        request = ChatRequest(messages=messages)
        result = await session.complete(request)
        assert result["choices"][0]["message"]["content"] == "hello"
        assert result["choices"][0]["finish_reason"] == "stop"
        turn = session.trace.turns[-1]
        assert turn.stop_reason == "stop"  # read off the SDK's SampledSequence, not a dict copy
        expected = render_prompt(session.renderer, request)
        assert list(turn.input_ids) == expected
        assert 1000 not in turn.input_ids
        assert turn.output_ids == (1000,)
        assert turn.logprobs == (-0.25,)
        call = session.policy.sampling_client.sample_async.call_args.kwargs
        assert call["prompt"].to_ints() == expected
        assert call["sampling_params"].temperature == 0.7
    assert len(session.trace.turns) == 4


@pytest.mark.asyncio
async def test_parser_failure_does_not_erase_sample(monkeypatch):
    session = make_session()

    def fail(_):
        raise ValueError("cannot parse")

    monkeypatch.setattr(session.renderer, "parse_response", fail)
    with pytest.raises(ValueError, match="cannot parse"):
        await session.complete(ChatRequest(messages=[{"role": "user", "content": "hi"}]))
    assert session.trace.turns[0].output_ids == (1000,)


@pytest.mark.asyncio
async def test_caps_and_stop_overrides():
    session = make_session()
    request = ChatRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=9, stop=[])
    await session.complete(request)
    params = session.policy.sampling_client.sample_async.call_args.kwargs["sampling_params"]
    assert params.max_tokens == 9
    assert params.stop == []
    session.max_datum_tokens = 1
    with pytest.raises(ValueError, match="budget"):
        await session.complete(request)
    assert len(session.trace.turns) == 1
    assert session.policy.sampling_client.sample_async.await_count == 1


@pytest.mark.asyncio
async def test_real_http_lifecycle_isolated_policies_and_validation():
    first, second = make_session(), make_session()
    server = SessionServer()
    async with server.serve() as port, httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as http:
        async with server.session(first) as path, server.session(second) as other:
            body = {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello", "tool_calls": None, "reasoning_content": None},
                    {"role": "user", "content": "continue"},
                ]
            }
            for extra in [{"n": 2}, {"stream": True}, {"max_tokens": 0}, {"chat_template_kwargs": {}}]:
                result = await http.post(f"{path}/v1/chat/completions", json={**body, **extra})
                assert result.status_code == 400, result.text
            result = await http.post(f"{path}/v1/chat/completions", json=body)
            assert result.status_code == 200, result.text
            assert len(first.trace.turns) == 1
            assert not second.trace.turns
            assert other != path
        assert (await http.post(f"{path}/v1/chat/completions", json=body)).status_code == 404
    assert not server.sessions
    with pytest.raises(ValueError, match="closed"):
        await first.complete(ChatRequest(**body))


@pytest.mark.asyncio
async def test_session_close_waits_for_inflight_sample():
    session = make_session()
    original = session.policy.sampling_client.sample_async.return_value
    started, release = asyncio.Event(), asyncio.Event()

    async def sample(**kwargs):
        started.set()
        await release.wait()
        return original

    session.policy.sampling_client.sample_async.side_effect = sample
    server = SessionServer()
    context = server.session(session)
    await context.__aenter__()
    request = asyncio.create_task(session.complete(ChatRequest(messages=[{"role": "user", "content": "hi"}])))
    await started.wait()
    closing = asyncio.create_task(context.__aexit__(None, None, None))
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await request
    await closing
    assert session.closed
    assert len(session.trace.turns) == 1
