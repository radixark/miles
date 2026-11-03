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
    FORCE_FINAL_TEXT,
    PARSE_FAILURE_REQUEST_ARGS,
    PARSE_FAILURE_USER_TEXT,
    DriverAction,
    fixed_template_append_roles,
    run_agent,
    select_schedule,
)
from miles.utils.types import Sample


def test_all_role_schedule_places_assistant_input_last():
    schedule = select_schedule(VALID_APPEND_ROLES, cycles=2)

    assert schedule[-1] is DriverAction.ASSISTANT_INPUT
    assert schedule.count(DriverAction.ASSISTANT_INPUT) == 1
    assert DriverAction.ROLLBACK in schedule[:-1]


def test_model_config_uses_family_fixed_template_capability():
    cfg = ModelConfig(
        model_name="model",
        reasoning_parser="reasoning",
        tool_call_parser="tool",
        tito_model="qwen3",
    )

    assert fixed_template_append_roles(cfg.tito_model) == VALID_APPEND_ROLES


def test_assistant_input_appends_two_text_messages_then_user(monkeypatch):
    calls = []

    class FakeAsyncClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    async def fake_chat(client, base_url, messages, request_kwargs, *, label):
        calls.append(deepcopy(messages))
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"generated response {len(calls)}",
                        "tool_calls": [],
                    }
                }
            ]
        }

    monkeypatch.setattr(session_verify_agent.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(session_verify_agent, "_chat", fake_chat)
    monkeypatch.setattr(
        session_verify_agent,
        "select_schedule",
        lambda allowed_roles, *, cycles: [DriverAction.ASSISTANT_INPUT],
    )

    result = asyncio.run(
        run_agent(
            "http://session",
            prompt=None,
            request_kwargs={},
            metadata={"tito_model": "qwen3"},
        )
    )

    assert [message["role"] for message in calls[1]] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "assistant",
        "user",
    ]
    assert [message["content"] for message in calls[1][-3:-1]] == list(ASSISTANT_INPUT_TEXTS)
    assert calls[1][-1] == {"role": "user", "content": ASSISTANT_INPUT_FOLLOWUP_TEXT}
    assert result["driver_events"] == ["initial", "append_assistant"]
    assert result["assistant_input_count"] == 2
    assert result["user_count"] == 1


def test_chat_complete_retries_length_with_distinct_seed(monkeypatch):
    attempts = []
    responses = iter(
        [
            {"choices": [{"finish_reason": "length"}]},
            {"choices": [{"finish_reason": "stop"}]},
        ]
    )

    async def fake_chat(client, base_url, messages, request_kwargs, *, label):
        attempts.append(request_kwargs)
        return next(responses)

    monkeypatch.setattr(session_verify_agent, "_chat", fake_chat)

    response = asyncio.run(
        session_verify_agent._chat_complete(
            None,
            "http://session",
            [],
            {"seed": 7},
            label="turn",
        )
    )

    assert response["choices"][0]["finish_reason"] == "stop"
    assert [kwargs["seed"] for kwargs in attempts] == [7, 1_000_007]


def test_minimax_schedule_excludes_system_and_keeps_assistant_rollback():
    roles = fixed_template_append_roles("minimax_m27")
    schedule = select_schedule(roles, cycles=1)

    assert roles == ("tool", "user", "assistant")
    assert DriverAction.SYSTEM_REMINDER not in schedule
    assert schedule[-1] is DriverAction.ASSISTANT_INPUT
    assert DriverAction.ROLLBACK in schedule[:-1]


def test_qwen35_schedule_excludes_system_and_keeps_assistant_rollback():
    roles = fixed_template_append_roles("qwen35")
    schedule = select_schedule(roles, cycles=1)

    assert roles == ("tool", "user", "assistant")
    assert DriverAction.SYSTEM_REMINDER not in schedule
    assert schedule[-1] is DriverAction.ASSISTANT_INPUT
    assert DriverAction.ROLLBACK in schedule[:-1]


def test_assistant_input_generated_response_is_rolled_back_and_regenerated(monkeypatch):
    calls = []

    class FakeAsyncClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    async def fake_chat(client, base_url, messages, request_kwargs, *, label):
        calls.append(deepcopy(messages))
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"generated response {len(calls)}",
                        "tool_calls": [],
                    }
                }
            ]
        }

    monkeypatch.setattr(session_verify_agent.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(session_verify_agent, "_chat", fake_chat)
    monkeypatch.setattr(
        session_verify_agent,
        "select_schedule",
        lambda allowed_roles, *, cycles: [DriverAction.ASSISTANT_INPUT, DriverAction.ROLLBACK],
    )

    result = asyncio.run(
        run_agent(
            "http://session",
            prompt=None,
            request_kwargs={},
            metadata={"tito_model": "qwen3"},
        )
    )

    injected_request = calls[1]
    rollback_request = calls[2]

    assert injected_request[-1] == {"role": "user", "content": ASSISTANT_INPUT_FOLLOWUP_TEXT}
    # The rollback retries the exact injected request: the popped generated
    # response is gone, the injected assistants stay as prompt history.
    assert rollback_request == injected_request
    assert all(message.get("content") != "generated response 2" for message in rollback_request)
    assert result["driver_events"] == ["initial", "append_assistant", "rollback"]
    assert result["assistant_input_count"] == 2
    assert result["user_count"] == 1
    assert result["rollback_count"] == 1


def test_unanswered_tool_call_is_answered_before_force_final(monkeypatch):
    calls = []
    tool_call = {
        "id": "call_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
    }

    class FakeAsyncClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    async def fake_chat(client, base_url, messages, request_kwargs, *, label):
        calls.append(deepcopy(messages))
        # The initial turn and the turn after the tool result both call the tool.
        tool_calls = [tool_call] if len(calls) <= 2 else []
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}}]}

    monkeypatch.setattr(session_verify_agent.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(session_verify_agent, "_chat", fake_chat)
    monkeypatch.setattr(
        session_verify_agent,
        "select_schedule",
        lambda allowed_roles, *, cycles: [DriverAction.TOOL_RESULT, DriverAction.FORCE_FINAL],
    )

    result = asyncio.run(
        run_agent(
            "http://session",
            prompt=None,
            request_kwargs={},
            metadata={"tito_model": "qwen3"},
        )
    )

    force_final_request = calls[2]
    assert [message["role"] for message in force_final_request] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "user",
    ]
    assert force_final_request[-2]["tool_call_id"] == "call_0"
    assert force_final_request[-1] == {"role": "user", "content": FORCE_FINAL_TEXT}
    assert result["driver_events"] == ["initial", "append_tool", "append_tool", "force_final"]
    assert result["tool_result_count"] == 2


def test_schedule_places_parse_failure_right_before_force_final():
    schedule = select_schedule(VALID_APPEND_ROLES, cycles=2)

    assert schedule.count(DriverAction.PARSE_FAILURE) == 1
    force_final = schedule.index(DriverAction.FORCE_FINAL)
    assert schedule[force_final - 1] is DriverAction.PARSE_FAILURE


def test_schedule_without_user_role_has_no_parse_failure():
    schedule = select_schedule(("tool",), cycles=1)

    assert DriverAction.PARSE_FAILURE not in schedule
    assert DriverAction.FORCE_FINAL not in schedule


def _run_parse_failure_schedule(monkeypatch, parse_failure_reply):
    calls = []

    class FakeAsyncClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    async def fake_chat(client, base_url, messages, request_kwargs, *, label):
        calls.append((deepcopy(messages), dict(request_kwargs)))
        if messages[-1]["content"] == PARSE_FAILURE_USER_TEXT:
            return {"choices": [{"message": parse_failure_reply}]}
        return {"choices": [{"message": {"role": "assistant", "content": "Beijing is sunny."}}]}

    monkeypatch.setattr(session_verify_agent.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(session_verify_agent, "_chat", fake_chat)
    monkeypatch.setattr(
        session_verify_agent,
        "select_schedule",
        lambda allowed_roles, *, cycles: [DriverAction.PARSE_FAILURE, DriverAction.FORCE_FINAL],
    )
    result = asyncio.run(
        run_agent(
            "http://session",
            prompt=None,
            request_kwargs={"temperature": 1.0},
            metadata={"tito_model": "qwen3"},
        )
    )
    return calls, result


def test_parse_failure_reply_is_unparsed_and_followed_by_user(monkeypatch):
    raw_call = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Hangzhou"}}\n</tool_call>'
    unparsed = {"role": "assistant", "content": raw_call}

    calls, result = _run_parse_failure_schedule(monkeypatch, unparsed)

    (_, initial_kwargs), (_, parse_failure_kwargs), (force_final_messages, force_final_kwargs) = calls
    assert parse_failure_kwargs == {"temperature": 1.0, **PARSE_FAILURE_REQUEST_ARGS}
    assert initial_kwargs == force_final_kwargs == {"temperature": 1.0}
    assert force_final_messages[-3:] == [
        {"role": "user", "content": PARSE_FAILURE_USER_TEXT},
        unparsed,
        {"role": "user", "content": FORCE_FINAL_TEXT},
    ]
    assert result["driver_events"] == ["initial", "parse_failure", "force_final"]


def test_parse_failure_rejects_a_reply_with_parsed_tool_calls(monkeypatch):
    parsed = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
    }

    with pytest.raises(AssertionError, match="tool_choice='none' still returned tool_calls"):
        _run_parse_failure_schedule(monkeypatch, parsed)


def _qwen3_verified_sample(mismatches):
    return Sample(
        metadata={
            "driver_events": [
                "initial",
                "append_tool",
                "append_user",
                "parse_failure",
                "append_system",
                "append_assistant",
                "rollback",
            ],
            "tito_session_mismatch": mismatches,
        }
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
    returned_sample = _qwen3_verified_sample(
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
    input_value = SimpleNamespace(
        sample=Sample(),
        args=SimpleNamespace(
            tito_model="qwen3",
            session_verify_cycles=3,
            tool_call_failure_mode="rollback",
        ),
    )

    output = asyncio.run(session_verify_agent.generate(input_value))

    assert output.samples == [returned_sample]
    [metric] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert metric["hard_mismatch_count"] == 1
    assert metric["hard_mismatch_types"] == ["special_token_count"]
    assert metric["hard_mismatch_example"] == {
        "type": "special_token_count",
        "segment_index": -1,
        "detail": "segment count differs: expected 99, got 97",
    }
    assert metric["had_assistant_mismatch"] is True
    assert writes == [(json.dumps(metric) + "\n").encode()]


def test_hard_mismatch_still_raises_without_metrics_sidecar(monkeypatch):
    monkeypatch.delenv("MILES_SESSION_VERIFY_METRICS_PATH", raising=False)
    sample = _qwen3_verified_sample([{"type": "non_assistant_text"}])

    with pytest.raises(AssertionError, match="forbidden mismatches"):
        session_verify_agent._verify_tito_samples(
            [sample],
            [sample.metadata["driver_events"]],
            allowed_roles=list(VALID_APPEND_ROLES),
        )


def test_run_agent_journals_assertion_before_retry(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))

    def reject_template(_tito_model):
        raise AssertionError("journal me")

    monkeypatch.setattr(session_verify_agent, "fixed_template_append_roles", reject_template)

    with pytest.raises(AssertionError, match="journal me"):
        asyncio.run(
            run_agent(
                "http://session",
                prompt=None,
                request_kwargs={},
                metadata={"tito_model": "qwen3"},
            )
        )

    [record] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert record["verification_error"]["stage"] == "session_verify_agent.run_agent"
    assert record["verification_error"]["message"] == "journal me"


def test_generate_journals_coverage_failure_after_hard_mismatch(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    sample = _qwen3_verified_sample([{"type": "special_token_count", "detail": "count differs"}])
    sample.metadata["driver_events"].remove("rollback")

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[sample])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)
    input_value = SimpleNamespace(
        sample=Sample(),
        args=SimpleNamespace(
            tito_model="qwen3",
            session_verify_cycles=3,
            tool_call_failure_mode="rollback",
        ),
    )

    with pytest.raises(AssertionError, match="missing required driver events"):
        asyncio.run(session_verify_agent.generate(input_value))

    sample_record, error_record = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert sample_record["hard_mismatch_count"] == 1
    assert error_record["verification_error"]["stage"] == "session_verify_agent.generate"
