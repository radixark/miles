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


def _verified_sample(mismatches, *, expectations=()):
    return Sample(
        metadata={
            "driver_events": ["initial", "append_tool", "append_user"],
            EXPECTATIONS_KEY: list(expectations),
            "tito_session_mismatch": mismatches,
        }
    )


def _generate_input(kind: SessionVerifyKind, tito_model="qwen3"):
    return SimpleNamespace(
        sample=Sample(metadata={SESSION_VERIFY_KIND_KEY: str(kind)}),
        args=SimpleNamespace(tito_model=tito_model, session_verify_cycles=3),
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


def test_generate_journals_a_spec_sample_missing_probes(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    sample = _verified_sample([], expectations=[{"step": "user after tool call", "expected_status": 200}])
    sample.metadata[EXPECTATIONS_KEY][0]["actual_status"] = 200

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[sample])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)

    with pytest.raises(AssertionError, match="recorded probes"):
        asyncio.run(session_verify_agent.generate(_generate_input(SessionVerifyKind.SPEC)))
    _, error_record = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert error_record["verification_error"]["stage"] == "session_verify_agent.generate"


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
