from types import SimpleNamespace

import pytest

import miles.rollout.generate_hub.agentic_tool_call as agentic_tool_call
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.session.request_overrides import SESSION_REQUEST_OVERRIDE_KEYS
from miles.rollout.session.samples.codec import SamplesReply
from miles.utils.types import Sample


class _Tracer:
    session_id = "sid-1"
    session_server_id = "127.0.0.1:12345"
    session_server_instance_id = None
    base_url = "http://127.0.0.1:12345/sessions/sid-1"

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.agent_metadata = None

    async def collect_samples(self, input_sample, *, max_seq_len, agent_metadata=None):
        self.agent_metadata = agent_metadata
        if self.error is not None:
            raise self.error
        return self.reply


def _generate_input(*, sampling_params: dict | None = None, **args_kwargs) -> GenerateFnInput:
    args = SimpleNamespace(
        session_server_ip="127.0.0.1",
        session_server_ports=[12345],
        custom_agent_function_path="test.fake_agent",
        max_seq_len=None,
        use_session_server="v2",
        **args_kwargs,
    )
    state = SimpleNamespace(args=args)
    sample = Sample(
        group_index=3,
        index=7,
        prompt=[{"role": "user", "content": "hello"}],
        label="label",
        metadata={"source": "test"},
    )
    return GenerateFnInput(
        state=state,
        sample=sample,
        sampling_params=sampling_params or {},
        evaluation=False,
    )


async def _fake_agent(**kwargs):
    return {"agent_result": "done"}


def _patch_agent(monkeypatch, tracer):
    async def fake_create(args, *, request_overrides=None):
        tracer.request_overrides = request_overrides
        return tracer

    monkeypatch.setattr(agentic_tool_call.OpenAIEndpointTracer, "create", fake_create)
    monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: _fake_agent)


@pytest.mark.asyncio
async def test_success_returns_list_and_forwards_agent_metadata(monkeypatch):
    sample = Sample(status=Sample.Status.COMPLETED, response="done", response_length=1, tokens=[1])
    tracer = _Tracer(SamplesReply(samples=[sample], session_metadata={}, empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    output = await agentic_tool_call.generate(_generate_input())

    assert output.samples == [sample]
    assert tracer.agent_metadata == {"agent_result": "done"}


@pytest.mark.asyncio
async def test_sampling_params_are_pinned_and_forwarded_to_agent(monkeypatch):
    sample = Sample(status=Sample.Status.COMPLETED, response="done", response_length=1, tokens=[1])
    tracer = _Tracer(SamplesReply(samples=[sample], session_metadata={}, empty_reason=None))
    _patch_agent(monkeypatch, tracer)
    seen: dict = {}

    async def capture_agent(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: capture_agent)
    sampling_params = {
        "max_new_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "sampling_seed": 42,
    }

    await agentic_tool_call.generate(_generate_input(sampling_params=sampling_params))

    expected = {
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "seed": 42,
    }
    assert tracer.request_overrides == expected
    assert seen["request_kwargs"] == expected


def test_sampling_params_exclude_chat_fields_rejected_by_sessions() -> None:
    unsupported_chat_fields = (
        set(agentic_tool_call.ChatCompletionRequest.model_fields)
        - SESSION_REQUEST_OVERRIDE_KEYS
        - {"model", "messages"}
    )
    assert unsupported_chat_fields
    sampling_params = {key: True for key in unsupported_chat_fields}
    sampling_params["temperature"] = 0.7

    request_kwargs = agentic_tool_call.build_chat_request_kwargs(sampling_params)

    assert request_kwargs == {"temperature": 0.7}
    assert set(request_kwargs) <= SESSION_REQUEST_OVERRIDE_KEYS


def test_session_override_allowlist_is_supported_by_sglang() -> None:
    assert SESSION_REQUEST_OVERRIDE_KEYS <= set(agentic_tool_call.ChatCompletionRequest.model_fields)


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_reason", ["no_records", "all_truncated"])
async def test_empty_reply_returns_aborted_list(monkeypatch, empty_reason):
    tracer = _Tracer(SamplesReply(samples=[], session_metadata={}, empty_reason=empty_reason))
    _patch_agent(monkeypatch, tracer)
    generate_input = _generate_input()

    output = await agentic_tool_call.generate(generate_input)

    assert isinstance(output.samples, list)
    assert len(output.samples) == 1
    assert output.samples[0] is not generate_input.sample
    assert output.samples[0].status == Sample.Status.ABORTED


@pytest.mark.asyncio
async def test_collection_error_propagates(monkeypatch):
    tracer = _Tracer(error=RuntimeError("samples unavailable"))
    _patch_agent(monkeypatch, tracer)

    with pytest.raises(RuntimeError, match="samples unavailable"):
        await agentic_tool_call.generate(_generate_input())
