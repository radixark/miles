from types import SimpleNamespace

import pytest

import miles.rollout.generate_hub.agentic_tool_call as agentic_tool_call
from miles.ray.rollout.rollout_data_conversion import validate_compact_rollout_ids
from miles.rollout.base_types import GenerateFnInput
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


def _generate_input(**args_kwargs) -> GenerateFnInput:
    args = SimpleNamespace(
        session_server_addrs=["127.0.0.1:12345"],
        custom_agent_function_path="test.fake_agent",
        max_seq_len=None,
        partial_rollout=False,
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
    return GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False)


async def _fake_agent(**kwargs):
    return {"agent_result": "done"}


def _patch_agent(monkeypatch, tracer):
    async def fake_create(args):
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
    assert output.samples[0].rollout_id is None
    assert tracer.agent_metadata == {"agent_result": "done"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("input_rollout_id", "expected_rollout_id"), [(None, 7), (11, 11)])
async def test_success_assigns_shared_rollout_id_to_v2_leaves(monkeypatch, input_rollout_id, expected_rollout_id):
    leaves = [
        Sample(status=Sample.Status.COMPLETED, response="one", response_length=1, tokens=[1]),
        Sample(status=Sample.Status.COMPLETED, response="two", response_length=1, tokens=[2]),
    ]
    tracer = _Tracer(SamplesReply(samples=leaves, session_metadata={}, empty_reason=None))
    _patch_agent(monkeypatch, tracer)
    generate_input = _generate_input()
    generate_input.sample.rollout_id = input_rollout_id

    output = await agentic_tool_call.generate(generate_input)

    assert [sample.rollout_id for sample in output.samples] == [expected_rollout_id] * 2
    validate_compact_rollout_ids([[output.samples]])


@pytest.mark.asyncio
async def test_v2_requires_input_rollout_identity(monkeypatch):
    leaves = [
        Sample(status=Sample.Status.COMPLETED, response="one", response_length=1, tokens=[1]),
        Sample(status=Sample.Status.COMPLETED, response="two", response_length=1, tokens=[2]),
    ]
    tracer = _Tracer(SamplesReply(samples=leaves, session_metadata={}, empty_reason=None))
    _patch_agent(monkeypatch, tracer)
    generate_input = _generate_input()
    generate_input.sample.index = None
    generate_input.sample.rollout_id = None

    with pytest.raises(AssertionError, match="require input Sample.rollout_id or Sample.index"):
        await agentic_tool_call.generate(generate_input)


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


_ADDRS_ATTR_ABSENT = object()


class TestSessionServerAddrsValidation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("addrs", [_ADDRS_ATTR_ABSENT, None, []], ids=["absent", "none", "empty"])
    async def test_empty_session_server_addrs_is_rejected(self, monkeypatch, addrs):
        """generate() raises the documented AssertionError when session_server_addrs is absent, null or empty, without creating a tracer."""
        created_for: list[object] = []

        async def fake_create(args):
            created_for.append(args)
            return _Tracer(SamplesReply(samples=[], session_metadata={}, empty_reason="no_records"))

        monkeypatch.setattr(agentic_tool_call.OpenAIEndpointTracer, "create", fake_create)
        monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: _fake_agent)

        generate_input = _generate_input()
        if addrs is _ADDRS_ATTR_ABSENT:
            del generate_input.args.session_server_addrs
        else:
            generate_input.args.session_server_addrs = addrs

        with pytest.raises(AssertionError, match="requires session_server_addrs"):
            await agentic_tool_call.generate(generate_input)

        assert created_for == []


@pytest.mark.asyncio
async def test_collection_error_propagates(monkeypatch):
    tracer = _Tracer(error=RuntimeError("samples unavailable"))
    _patch_agent(monkeypatch, tracer)

    with pytest.raises(RuntimeError, match="samples unavailable"):
        await agentic_tool_call.generate(_generate_input())
