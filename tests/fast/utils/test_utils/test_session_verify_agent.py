import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig

from miles.rollout.base_types import GenerateFnOutput
from miles.utils.chat_template_utils.tito_tokenizer import VALID_APPEND_ROLES
from miles.utils.test_utils import session_verify_agent
from miles.utils.test_utils.session_verify_agent import (
    ASSISTANT_INPUT_FOLLOWUP_TEXT,
    ASSISTANT_INPUT_TEXTS,
    EXPECTATIONS_KEY,
    NO_TOOL_CALL_REQUEST_ARGS,
    NO_TOOL_CALL_USER_TEXT,
    SESSION_VERIFY_KIND_KEY,
    SYSTEM_REMINDER_TEXT,
    UNFINISHED_KEY,
    USER_FOLLOWUP_TEXTS,
    SessionVerifyKind,
    fixed_template,
    fixed_template_append_roles,
    run_agent,
    spec_probe_steps,
)
from miles.utils.types import Sample


class _FakeAsyncClient:
    def __init__(self, *, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakeSession:
    """A session server for one family: refuses what its continuation roles refuse.

    The model calls the tool on a city weather question and answers in text otherwise.
    """

    def __init__(self, tito_model: str, *, refuse=None):
        self.fixed = fixed_template(tito_model)
        self.refuse = refuse
        self.requests = []
        self.calls = 0

    def _reply(self, messages):
        last = messages[-1]
        if last["role"] == "user" and "weather in" in last["content"]:
            self.calls += 1
            call = {
                "id": f"call_{self.calls}",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "x"}'},
            }
            return {"role": "assistant", "content": "", "tool_calls": [call]}
        return {"role": "assistant", "content": f"answer {len(self.requests)}"}

    def _refused(self, messages):
        if self.refuse is not None:
            return self.refuse(messages)
        previous, appended = messages[-2], messages[-1]
        return previous["role"] == "assistant" and appended["role"] not in self.fixed.roles_after(previous)

    async def chat(self, client, base_url, messages, request_kwargs, *, label):
        self.requests.append(("chat", deepcopy(messages), dict(request_kwargs)))
        return {"choices": [{"message": self._reply(messages), "finish_reason": "stop"}]}

    async def chat_status(self, client, base_url, messages, request_kwargs, *, label):
        self.requests.append(("probe", deepcopy(messages), dict(request_kwargs)))
        if self._refused(messages):
            return 400, None
        return 200, {"choices": [{"message": self._reply(messages), "finish_reason": "stop"}]}


def _run(monkeypatch, session: _FakeSession, kind: SessionVerifyKind, tito_model: str, **metadata):
    monkeypatch.setattr(session_verify_agent.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(session_verify_agent, "_chat", session.chat)
    monkeypatch.setattr(session_verify_agent, "_chat_status", session.chat_status)
    return asyncio.run(
        run_agent(
            "http://session",
            prompt=None,
            request_kwargs={"temperature": 1.0},
            metadata={"tito_model": tito_model, SESSION_VERIFY_KIND_KEY: str(kind), **metadata},
        )
    )


def test_model_config_uses_family_fixed_template_capability():
    cfg = ModelConfig(model_name="model", reasoning_parser="reasoning", tool_call_parser="tool", tito_model="qwen3")

    assert fixed_template_append_roles(cfg.tito_model) == VALID_APPEND_ROLES


def test_normal_rollout_answers_calls_with_tools_and_answers_with_questions(monkeypatch):
    session = _FakeSession("qwen3")

    result = _run(monkeypatch, session, SessionVerifyKind.NORMAL, "qwen3", session_verify_cycles=2)

    assert result["driver_events"] == ["initial", "append_tool", "append_user", "append_tool", "append_user"]
    assert result[EXPECTATIONS_KEY] == []
    final_messages = session.requests[-1][1]
    assert [message["role"] for message in final_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert all(kind == "chat" for kind, _, _ in session.requests)
    assert result["rollback_count"] == result["system_count"] == result["assistant_input_count"] == 0


def test_spec_rollout_on_a_lenient_family_gets_every_probe_accepted(monkeypatch):
    session = _FakeSession("qwen3")

    result = _run(monkeypatch, session, SessionVerifyKind.SPEC, "qwen3")

    assert result[EXPECTATIONS_KEY] == [
        {"step": step, "expected_status": 200, "actual_status": 200} for step in spec_probe_steps(session.fixed)
    ]
    assert [step for step in spec_probe_steps(session.fixed)] == [
        "user after tool call",
        "system after tool call",
        "assistant after tool call",
        "tool after no tool call",
    ]
    events = result["driver_events"]
    assert events.count("rollback") == 2
    assert events[-2:] == ["append_system", "append_assistant"]
    assert "no_tool_call" in events
    probes = [messages for kind, messages, _ in session.requests if kind == "probe"]
    assert [messages[-1]["role"] for messages in probes] == ["user", "system", "assistant", "tool"]
    assert [bool(messages[-2].get("tool_calls")) for messages in probes] == [True, True, True, False]
    assert probes[-1][-1] == {"role": "tool", "content": '{"result": "unavailable"}', "tool_call_id": "none"}
    final = session.requests[-1][1]
    assert [message["content"] for message in final[-3:-1]] == list(ASSISTANT_INPUT_TEXTS)
    assert final[-1] == {"role": "user", "content": ASSISTANT_INPUT_FOLLOWUP_TEXT}
    assert {"role": "system", "content": SYSTEM_REMINDER_TEXT} in final
    # The questions that rebuild a call after the system and assistant probes name new cities.
    questions = [message["content"] for message in final if message["role"] == "user"]
    assert questions[1:4] == list(USER_FOLLOWUP_TEXTS)


def test_spec_rollout_expects_400_where_the_spec_refuses(monkeypatch):
    session = _FakeSession("deepseekv4")

    result = _run(monkeypatch, session, SessionVerifyKind.SPEC, "deepseekv4")

    assert result[EXPECTATIONS_KEY] == [
        {"step": "user after tool call", "expected_status": 400, "actual_status": 400},
        {"step": "assistant after tool call", "expected_status": 400, "actual_status": 400},
        {"step": "tool after no tool call", "expected_status": 400, "actual_status": 400},
    ]
    # Both refused probes left the call pending, so the next request answers that same call.
    probes = [messages for kind, messages, _ in session.requests if kind == "probe"]
    first_call = probes[0][-2]
    chats = [messages for kind, messages, _ in session.requests if kind == "chat"]
    assert chats[1][-2:] == [
        first_call,
        {"role": "tool", "content": chats[1][-1]["content"], "tool_call_id": "call_1"},
    ]
    assert "append_system" not in result["driver_events"]
    assert result["driver_events"][-1] == "append_assistant"


def test_spec_rollout_records_a_status_the_spec_did_not_expect_and_continues(monkeypatch):
    # The server wrongly accepts every continuation.
    session = _FakeSession("deepseekv4", refuse=lambda messages: False)

    result = _run(monkeypatch, session, SessionVerifyKind.SPEC, "deepseekv4")

    assert [(e["expected_status"], e["actual_status"]) for e in result[EXPECTATIONS_KEY]] == [(400, 200)] * 3
    assert result["driver_events"][-1] == "append_assistant"


def test_no_tool_call_turn_uses_the_guard_and_rejects_parsed_calls(monkeypatch):
    session = _FakeSession("qwen3")
    original_reply = session._reply

    def reply(messages):
        if messages[-1].get("content") == NO_TOOL_CALL_USER_TEXT:
            return {"role": "assistant", "content": "", "tool_calls": [{"id": "call_x"}]}
        return original_reply(messages)

    session._reply = reply

    with pytest.raises(AssertionError, match="tool_choice='none' still returned tool_calls"):
        _run(monkeypatch, session, SessionVerifyKind.SPEC, "qwen3")
    no_call_request = session.requests[-1]
    assert no_call_request[1][-1]["content"] == NO_TOOL_CALL_USER_TEXT
    assert no_call_request[2] == {"temperature": 1.0, **NO_TOOL_CALL_REQUEST_ARGS}


def test_spec_rollout_resamples_a_turn_that_must_call_a_tool(monkeypatch):
    session = _FakeSession("qwen3")
    original_reply = session._reply
    seeds = []

    def reply(messages):
        if len(messages) == 2 and len(seeds) == 1:
            return {"role": "assistant", "content": "no call yet"}
        return original_reply(messages)

    async def chat(client, base_url, messages, request_kwargs, *, label):
        if len(messages) == 2:
            seeds.append(request_kwargs.get("seed"))
        return await _FakeSession.chat(session, client, base_url, messages, request_kwargs, label=label)

    session._reply = reply
    monkeypatch.setattr(session_verify_agent.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(session_verify_agent, "_chat", chat)
    monkeypatch.setattr(session_verify_agent, "_chat_status", session.chat_status)
    asyncio.run(
        run_agent(
            "http://session",
            prompt=None,
            request_kwargs={"seed": 7},
            metadata={"tito_model": "qwen3", SESSION_VERIFY_KIND_KEY: "spec"},
        )
    )

    assert seeds == [7, 1_000_007]


def test_spec_rollout_is_unfinished_when_the_model_never_stops_calling_the_tool(monkeypatch):
    """The sample stops at that step and keeps the probes it sent; the agent does not raise."""
    session = _FakeSession("qwen3")

    async def chat(client, base_url, messages, request_kwargs, *, label):
        if label == "Tool result" or label.startswith("tool round"):
            call = {"id": f"again_{len(messages)}", "type": "function", "function": {"name": "get_weather"}}
            message = {"role": "assistant", "content": "", "tool_calls": [call]}
            return {"choices": [{"message": message, "finish_reason": "stop"}]}
        return await _FakeSession.chat(session, client, base_url, messages, request_kwargs, label=label)

    monkeypatch.setattr(session_verify_agent.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(session_verify_agent, "_chat", chat)
    monkeypatch.setattr(session_verify_agent, "_chat_status", session.chat_status)

    result = asyncio.run(
        run_agent(
            "http://session",
            prompt=None,
            request_kwargs={},
            metadata={"tito_model": "qwen3", SESSION_VERIFY_KIND_KEY: "spec"},
        )
    )

    assert result[UNFINISHED_KEY] == "the model still calls a tool after 3 answered rounds"
    steps = spec_probe_steps(fixed_template("qwen3"))
    assert [e["step"] for e in result[EXPECTATIONS_KEY]] == steps[:-1]
    assert "no_tool_call" not in result["driver_events"]


def test_spec_rollout_is_unfinished_when_the_model_never_calls_the_tool(monkeypatch):
    session = _FakeSession("qwen3")
    session._reply = lambda messages: {"role": "assistant", "content": "It is sunny."}

    result = _run(monkeypatch, session, SessionVerifyKind.SPEC, "qwen3")

    assert result[UNFINISHED_KEY].startswith("Initial exhausted 4 resamples")
    assert result[EXPECTATIONS_KEY] == []
    assert result["driver_events"] == []


def test_a_finished_spec_rollout_is_not_marked_unfinished(monkeypatch):
    result = _run(monkeypatch, _FakeSession("qwen3"), SessionVerifyKind.SPEC, "qwen3")

    assert result[UNFINISHED_KEY] is None


def test_spec_rollout_needs_user_after_an_answer():
    fixed = session_verify_agent.FixedTemplate(
        allowed_append_roles=frozenset({"tool", "user"}), roles_after_no_tool_call=frozenset({"tool"})
    )

    with pytest.raises(ValueError, match="needs 'user' after an assistant without tool calls"):
        session_verify_agent._assert_spec_rollout_supported("custom", fixed)


def test_run_agent_requires_the_rollout_kind():
    with pytest.raises(ValueError, match="session_verify_kind"):
        asyncio.run(run_agent("http://session", prompt=None, request_kwargs={}, metadata={"tito_model": "qwen3"}))


def test_chat_retries_length_with_distinct_seed(monkeypatch):
    attempts = []
    responses = iter(
        [
            {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "cut"}}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}]},
        ]
    )

    async def fake_chat(client, base_url, messages, request_kwargs, *, label):
        attempts.append(request_kwargs)
        return next(responses)

    monkeypatch.setattr(session_verify_agent, "_chat", fake_chat)
    driver = session_verify_agent._Driver(client=None, base_url="http://session", request_kwargs={"seed": 7})

    message = asyncio.run(driver.chat([], label="turn"))

    assert message["content"] == "done"
    assert [kwargs["seed"] for kwargs in attempts] == [7, 1_000_007]


def test_chat_fails_when_length_cuts_exhaust_the_resamples(monkeypatch):
    """Only a turn that needs a tool call leaves a sample unfinished; a turn cut every time fails."""

    async def fake_chat(client, base_url, messages, request_kwargs, *, label):
        return {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "cut"}}]}

    monkeypatch.setattr(session_verify_agent, "_chat", fake_chat)
    driver = session_verify_agent._Driver(client=None, base_url="http://session", request_kwargs={})

    with pytest.raises(AssertionError, match="turn exhausted 4 resamples"):
        asyncio.run(driver.chat([], label="turn"))


def test_chat_fails_when_a_turn_that_needs_a_tool_call_is_cut_every_time(monkeypatch):
    async def fake_chat(client, base_url, messages, request_kwargs, *, label):
        return {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "cut"}}]}

    monkeypatch.setattr(session_verify_agent, "_chat", fake_chat)
    driver = session_verify_agent._Driver(client=None, base_url="http://session", request_kwargs={})

    with pytest.raises(AssertionError, match="Initial exhausted 4 resamples"):
        asyncio.run(driver.chat([], label="Initial", need_tool_call=True))


def test_chat_reports_every_rejected_reply_when_resamples_run_out(monkeypatch):
    async def fake_chat(client, base_url, messages, request_kwargs, *, label):
        return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "It is sunny."}}]}

    monkeypatch.setattr(session_verify_agent, "_chat", fake_chat)
    driver = session_verify_agent._Driver(client=None, base_url="http://session", request_kwargs={})

    with pytest.raises(session_verify_agent._ModelBehaviorError, match="Tool call after system exhausted") as raised:
        asyncio.run(driver.chat([], label="Tool call after system", need_tool_call=True))

    attempts = session_verify_agent._MAX_RESAMPLES + 1
    assert str(raised.value).count("stop: 'It is sunny.'") == attempts


def _verified_sample(mismatches, *, expectations=(), unfinished=None):
    return Sample(
        metadata={
            "driver_events": ["initial", "append_tool", "append_user"],
            EXPECTATIONS_KEY: list(expectations),
            UNFINISHED_KEY: unfinished,
            "tito_session_mismatch": mismatches,
        }
    )


def _unfinished_spec_sample(tito_model="qwen3", *, probes: int | None = 1):
    steps = spec_probe_steps(fixed_template(tito_model))[:probes]
    expectations = [{"step": step, "expected_status": 200, "actual_status": 200} for step in steps]
    return _verified_sample([], expectations=expectations, unfinished="Tool call after user exhausted 4 resamples")


def _generate_input(kind: SessionVerifyKind, tito_model="qwen3", *, group_index=0, num_rollout=2):
    return SimpleNamespace(
        sample=Sample(group_index=group_index, metadata={SESSION_VERIFY_KIND_KEY: str(kind)}),
        args=SimpleNamespace(
            tito_model=tito_model, session_verify_cycles=3, rollout_batch_size=16, num_rollout=num_rollout
        ),
    )


def test_generate_records_hard_mismatch_without_dropping_sample(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    writes = []
    real_write = session_verify_agent.os.write

    def tracked_write(fd, payload):
        writes.append(payload)
        return real_write(fd, payload)

    monkeypatch.setattr(session_verify_agent.os, "write", tracked_write)
    returned_sample = _verified_sample(
        [
            {
                "type": "special_token_count",
                "segment_index": -1,
                "detail": "segment count differs: expected 99, got 97",
            },
            {
                "type": "assistant_text",
                "segment_index": 3,
                "expected_text": "expected",
                "actual_text": "actual",
            },
        ]
    )

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[returned_sample])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)

    output = asyncio.run(session_verify_agent.generate(_generate_input(SessionVerifyKind.NORMAL)))

    assert output.samples == [returned_sample]
    [metric] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert metric["kind"] == "normal"
    assert metric["expectations"] == []
    assert metric["hard_mismatch_count"] == 1
    assert metric["hard_mismatch_types"] == ["special_token_count"]
    assert metric["hard_mismatch_example"] == {
        "type": "special_token_count",
        "segment_index": -1,
        "detail": "segment count differs: expected 99, got 97",
    }
    assert metric["had_assistant_mismatch"] is True
    assert writes == [(json.dumps(metric) + "\n").encode()]


def test_generate_records_spec_expectations(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    steps = spec_probe_steps(fixed_template("deepseekv4"))
    expectations = [{"step": step, "expected_status": 400, "actual_status": 400} for step in steps]
    sample = _verified_sample([], expectations=expectations)

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[sample])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)

    asyncio.run(session_verify_agent.generate(_generate_input(SessionVerifyKind.SPEC, "deepseekv4")))

    [metric] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert metric["kind"] == "spec"
    assert metric["expectations"] == expectations


@pytest.mark.parametrize("probes", [0, 1, None], ids=["before-any-probe", "after-one-probe", "after-every-probe"])
def test_generate_records_an_unfinished_spec_sample_without_failing_the_run(monkeypatch, tmp_path, probes):
    """The sample carries the probes up to the step it stopped at; the runner gates how many stop."""
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    sample = _unfinished_spec_sample(probes=probes)

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[sample])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)

    asyncio.run(session_verify_agent.generate(_generate_input(SessionVerifyKind.SPEC)))

    [metric] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert metric["unfinished"] == "Tool call after user exhausted 4 resamples"
    assert [e["step"] for e in metric["expectations"]] == spec_probe_steps(fixed_template("qwen3"))[:probes]


def test_generate_journals_an_unfinished_spec_sample_whose_probes_are_out_of_order(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    sample = _unfinished_spec_sample()
    sample.metadata[EXPECTATIONS_KEY][0]["step"] = "tool after no tool call"

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[sample])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)

    asyncio.run(session_verify_agent.generate(_generate_input(SessionVerifyKind.SPEC)))

    _, error_record = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert "recorded probes" in error_record["verification_error"]["message"]


def test_generate_raises_for_an_unfinished_spec_sample_without_metrics_sidecar(monkeypatch):
    """No run gate counts unfinished samples there, so the sample fails and its group is retried."""
    monkeypatch.delenv("MILES_SESSION_VERIFY_METRICS_PATH", raising=False)

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[_unfinished_spec_sample()])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)

    with pytest.raises(AssertionError, match="spec sample 0 is unfinished: Tool call after user exhausted"):
        asyncio.run(session_verify_agent.generate(_generate_input(SessionVerifyKind.SPEC)))


def test_generate_journals_a_spec_sample_missing_probes(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    sample = _verified_sample([], expectations=[{"step": "user after tool call", "expected_status": 200}])
    sample.metadata[EXPECTATIONS_KEY][0]["actual_status"] = 200

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[sample])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)

    # No raise: a dropped group makes the rollout loop refill the step with the next rollout's prompts.
    output = asyncio.run(session_verify_agent.generate(_generate_input(SessionVerifyKind.SPEC)))

    assert output.samples == [sample]
    _, error_record = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert error_record["verification_error"]["stage"] == "session_verify_agent.generate"
    assert "recorded probes" in error_record["verification_error"]["message"]


@pytest.mark.parametrize(
    ("num_rollout", "group_index", "refilled"),
    [(2, 31, False), (2, 32, True), (1, 16, False)],
    ids=["last-planned-group", "first-refill-group", "single-step-refill"],
)
def test_generate_journals_a_prompt_past_the_planned_ones(monkeypatch, tmp_path, num_rollout, group_index, refilled):
    """Two steps of 16 prompt groups plan groups 0..31; a later group means a failed group was refilled.

    A single step runs one kind, so its refill is not an error.
    """
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[_verified_sample([])])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)

    input = _generate_input(SessionVerifyKind.NORMAL, group_index=group_index, num_rollout=num_rollout)
    asyncio.run(session_verify_agent.generate(input))

    errors = [
        record["verification_error"]["message"]
        for record in map(json.loads, metrics_path.read_text().splitlines())
        if "verification_error" in record
    ]
    assert [("lies past the 32 planned ones" in message) for message in errors] == [True] * refilled


def test_generate_keeps_a_prompt_past_the_planned_ones_without_metrics_sidecar(monkeypatch):
    """No raise: a raise drops the group, the loop refills again, and every refill lies past the plan."""
    monkeypatch.delenv("MILES_SESSION_VERIFY_METRICS_PATH", raising=False)
    sample = _verified_sample([])

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[sample])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)

    output = asyncio.run(session_verify_agent.generate(_generate_input(SessionVerifyKind.NORMAL, group_index=32)))

    assert output.samples == [sample]


@pytest.mark.parametrize("kind", [SessionVerifyKind.NORMAL, SessionVerifyKind.SPEC])
def test_generate_journals_a_sample_whose_agent_raised(monkeypatch, tmp_path, kind):
    """The base generate collects the samples of an agent that raised; they carry no agent metadata."""
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    sample = Sample(metadata={"tito_session_mismatch": []})

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[sample])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)

    asyncio.run(session_verify_agent.generate(_generate_input(kind)))

    _, error_record = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert error_record["verification_error"]["stage"] == "session_verify_agent.generate"
    assert "carries no agent metadata" in error_record["verification_error"]["message"]


def test_generate_raises_for_a_sample_without_agent_metadata_without_metrics_sidecar(monkeypatch):
    monkeypatch.delenv("MILES_SESSION_VERIFY_METRICS_PATH", raising=False)

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[Sample(metadata={"tito_session_mismatch": []})])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)

    with pytest.raises(AssertionError, match="carries no agent metadata"):
        asyncio.run(session_verify_agent.generate(_generate_input(SessionVerifyKind.NORMAL)))


def test_generate_raises_for_a_spec_sample_missing_probes_without_metrics_sidecar(monkeypatch):
    monkeypatch.delenv("MILES_SESSION_VERIFY_METRICS_PATH", raising=False)

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[_verified_sample([])])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)

    with pytest.raises(AssertionError, match="recorded probes"):
        asyncio.run(session_verify_agent.generate(_generate_input(SessionVerifyKind.SPEC)))


@pytest.mark.parametrize(
    "sample",
    [
        _verified_sample([{"type": "non_assistant_text"}]),
        _verified_sample([], expectations=[{"step": "s", "expected_status": 400, "actual_status": 200}]),
    ],
    ids=["hard-mismatch", "unmet-expectation"],
)
def test_sample_failure_still_raises_without_metrics_sidecar(monkeypatch, sample):
    monkeypatch.delenv("MILES_SESSION_VERIFY_METRICS_PATH", raising=False)

    with pytest.raises(AssertionError, match="forbidden mismatches"):
        session_verify_agent._verify_tito_samples([sample], kind=SessionVerifyKind.SPEC)


def test_run_agent_journals_assertion_before_retry(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))

    def reject_template(_tito_model):
        raise AssertionError("journal me")

    monkeypatch.setattr(session_verify_agent, "fixed_template", reject_template)

    with pytest.raises(AssertionError, match="journal me"):
        asyncio.run(
            run_agent(
                "http://session",
                prompt=None,
                request_kwargs={},
                metadata={"tito_model": "qwen3", SESSION_VERIFY_KIND_KEY: "normal"},
            )
        )

    [record] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert record["verification_error"]["stage"] == "session_verify_agent.run_agent"
    assert record["verification_error"]["message"] == "journal me"
