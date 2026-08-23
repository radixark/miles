import asyncio
import json
from types import SimpleNamespace

import pytest

from miles.rollout.base_types import GenerateFnOutput
from miles.utils.test_utils import anthropic_session_verify_agent
from miles.utils.types import Sample


class _Response:
    def __init__(self, body: dict, status_code: int = 200):
        self._body = body
        self.status_code = status_code
        self.text = json.dumps(body)

    def json(self) -> dict:
        return self._body


def _response_fixtures():
    tool_uses = [
        {
            "type": "tool_use",
            "id": f"call_weather_{index}",
            "name": "get_weather",
            "input": {"location": location},
        }
        for index, location in enumerate(("Beijing", "Shanghai", "London"), start=1)
    ]
    tool_bodies = [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": f"Think about turn {index}."},
                tool_use,
            ],
            "stop_reason": "tool_use",
        }
        for index, tool_use in enumerate(tool_uses, start=1)
    ]
    text_bodies = [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": f"Use result {index}."},
                {"type": "text", "text": f"Weather answer {index}."},
            ],
            "stop_reason": "end_turn",
        }
        for index in range(1, 4)
    ]
    bodies = [item for pair in zip(tool_bodies, text_bodies, strict=True) for item in pair]

    canonical_messages = []
    for response_index, body in enumerate(bodies):
        reasoning = next(block["thinking"] for block in body["content"] if block["type"] == "thinking")
        if response_index % 2 == 0:
            tool_use = tool_uses[response_index // 2]
            message = {
                "content": "",
                "reasoning_content": reasoning,
                "tool_calls": [
                    {
                        "id": tool_use["id"],
                        "type": "function",
                        "function": {
                            "name": tool_use["name"],
                            "arguments": json.dumps(tool_use["input"]),
                        },
                    }
                ],
            }
        else:
            text = next(block["text"] for block in body["content"] if block["type"] == "text")
            message = {"content": text, "reasoning_content": reasoning}
        canonical_messages.append({"role": "assistant", **message})
    return tool_uses, bodies, canonical_messages


def _snapshot(canonical_messages, *, include_system: bool):
    records = []
    history_roles = []
    for record_index, response_message in enumerate(canonical_messages):
        turn_index = record_index // 2
        if record_index % 2 == 0:
            if turn_index == 2 and include_system:
                history_roles.append("system")
            history_roles.append("user")
        else:
            history_roles.extend(["assistant", "tool"])

        prior_assistants = iter(canonical_messages[:record_index])
        request_messages = [
            dict(next(prior_assistants)) if role == "assistant" else {"role": role}
            for role in ["system", *history_roles]
        ]
        input_ids = list(range(1, record_index * 2 + 2))
        completion_id = record_index * 2 + 2
        records.append(
            {
                "path": "/v1/chat/completions",
                "request": {"input_ids": input_ids, "messages": request_messages},
                "response": {
                    "id": f"response_{record_index}",
                    "choices": [
                        {
                            "message": response_message,
                            "meta_info": {"output_token_logprobs": [[-0.1, completion_id, None]]},
                        }
                    ]
                },
            }
        )
        if record_index % 2 == 1:
            history_roles.append("assistant")

    final_ids = records[-1]["request"]["input_ids"] + [len(records) * 2]
    metadata = {
        "max_trim_tokens": 0,
        "accumulated_token_ids": final_ids,
        "tree": {
            "nodes": [
                {
                    "id": index,
                    "parent": None if index == 0 else index - 1,
                    "response_id": f"response_{index}",
                }
                for index in range(6)
            ],
            "leaves": [{"node_id": 5, "path_node_ids": [0, 1, 2, 3, 4, 5]}],
        },
    }
    return {"records": records, "metadata": metadata}


@pytest.mark.parametrize(
    ("tito_model", "route_supports", "expectation", "include_system"),
    [
        ("qwen3", True, "required", True),
        ("qwen35", True, "forbidden", False),
        ("qwen35", False, "forbidden", False),
    ],
)
def test_run_agent_runs_six_turns_and_checks_canonical_records(
    monkeypatch, tito_model, route_supports, expectation, include_system
):
    posted = []
    _, response_bodies, canonical_messages = _response_fixtures()
    snapshot = _snapshot(canonical_messages, include_system=include_system)

    class FakeAsyncClient:
        def __init__(self, *, timeout):
            assert timeout == 180

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, *, json):
            body = response_bodies[len(posted)]
            posted.append((url, json))
            return _Response(body)

        async def get(self, url):
            if url == "http://session/health":
                return _Response({"status": "ok", "anthropic_intermediate_system_supported": route_supports})
            assert url == "http://session"
            return _Response(snapshot)

    monkeypatch.setattr(anthropic_session_verify_agent.httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(
        anthropic_session_verify_agent.run_agent(
            "http://session",
            prompt=None,
            request_kwargs={"max_tokens": 2048, "temperature": 0.2, "stop": "<stop>"},
            metadata={
                "anthropic_model": "/models/test",
                "tito_model": tito_model,
                "anthropic_intermediate_system_expectation": expectation,
            },
        )
    )

    assert len(posted) == 6
    assert {url for url, _ in posted} == {"http://session/v1/messages"}
    assert [payload["tool_choice"] for _, payload in posted] == [
        {"type": "tool", "name": "get_weather"},
        {"type": "none"},
    ] * 3
    assert all(payload["model"] == "/models/test" for _, payload in posted)
    assert all(payload["max_tokens"] == 1024 for _, payload in posted)
    assert all(payload["stream"] is False for _, payload in posted)
    assert all(payload["stop_sequences"] == ["<stop>"] for _, payload in posted)
    assert posted[0][1]["tools"][0]["input_schema"]["required"] == ["location"]

    first_result = posted[1][1]["messages"][-1]["content"][0]
    assert first_result["type"] == "tool_result"
    assert isinstance(first_result["content"], str)
    second_result = posted[3][1]["messages"][-1]["content"][0]
    assert second_result["content"][0]["type"] == "text"
    third_result = posted[5][1]["messages"][-1]["content"][0]
    assert third_result["content"] == anthropic_session_verify_agent._TOOL_RECOVERY_TEXT
    assert "is_error" not in third_result
    tail_roles = [message["role"] for message in posted[4][1]["messages"][-2:]]
    assert ("system" in tail_roles) is include_system
    assert posted[4][1]["messages"][-1]["content"][0]["type"] == "text"
    assert posted[1][1]["messages"][1]["content"][0]["type"] == "thinking"

    expected_events = anthropic_session_verify_agent._expected_driver_events(include_system=include_system)
    assert result == {
        "endpoint": "anthropic",
        "driver_events": expected_events,
        "request_count": 6,
        "tool_use_count": 3,
        "tool_result_count": 3,
        "text_turn_count": 3,
        "tool_result_string_count": 2,
        "tool_result_list_count": 1,
        "intermediate_system_used": include_system,
    }


def test_canonical_records_allow_superseded_retry_leaves():
    tool_uses, _, canonical_messages = _response_fixtures()
    snapshot = _snapshot(canonical_messages, include_system=True)
    snapshot["records"][-1]["response"]["id"] = "response_7"
    snapshot["metadata"]["tree"] = {
        "nodes": [
            {
                "id": index,
                "parent": None if index == 0 else index - 1,
                "response_id": f"response_{index}",
            }
            for index in range(5)
        ]
        + [
            {"id": 5, "parent": 4, "response_id": "retry_1"},
            {"id": 6, "parent": 4, "response_id": "retry_2"},
            {"id": 7, "parent": 4, "response_id": "response_7"},
        ],
        "leaves": [
            {"node_id": 5, "path_node_ids": [0, 1, 2, 3, 4, 5]},
            {"node_id": 6, "path_node_ids": [0, 1, 2, 3, 4, 6]},
            {"node_id": 7, "path_node_ids": [0, 1, 2, 3, 4, 7]},
        ],
    }

    anthropic_session_verify_agent._assert_canonical_records(
        snapshot,
        [[tool_use] for tool_use in tool_uses],
        include_system=True,
    )


def test_post_complete_retries_max_tokens():
    responses = iter(
        [
            _Response(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "truncated"}],
                    "stop_reason": "max_tokens",
                }
            ),
            _Response(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "complete"}],
                    "stop_reason": "end_turn",
                }
            ),
        ]
    )
    posted = []

    class FakeAsyncClient:
        async def post(self, url, *, json):
            posted.append((url, json))
            return next(responses)

    body = asyncio.run(
        anthropic_session_verify_agent._post_complete(
            FakeAsyncClient(),
            "http://session/v1/messages",
            {"messages": []},
            label="turn",
            assert_response=anthropic_session_verify_agent._assert_anthropic_text_response,
        )
    )

    assert body["stop_reason"] == "end_turn"
    assert posted == [
        ("http://session/v1/messages", {"messages": []}),
        ("http://session/v1/messages", {"messages": []}),
    ]


def test_post_complete_retries_empty_text_response():
    responses = iter(
        [
            _Response(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                    "stop_reason": "end_turn",
                }
            ),
            _Response(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "complete"}],
                    "stop_reason": "end_turn",
                }
            ),
        ]
    )

    class FakeAsyncClient:
        async def post(self, url, *, json):
            return next(responses)

    body = asyncio.run(
        anthropic_session_verify_agent._post_complete(
            FakeAsyncClient(),
            "http://session/v1/messages",
            {"messages": []},
            label="turn",
            assert_response=anthropic_session_verify_agent._assert_anthropic_text_response,
        )
    )

    assert body["content"] == [{"type": "text", "text": "complete"}]


def test_post_complete_raises_after_bounded_incomplete_responses():
    class FakeAsyncClient:
        attempts = 0

        async def post(self, url, *, json):
            self.attempts += 1
            return _Response(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                    "stop_reason": "end_turn",
                }
            )

    client = FakeAsyncClient()
    with pytest.raises(AssertionError):
        asyncio.run(
            anthropic_session_verify_agent._post_complete(
                client,
                "http://session/v1/messages",
                {"messages": []},
                label="turn",
                assert_response=anthropic_session_verify_agent._assert_anthropic_text_response,
            )
        )

    assert client.attempts == anthropic_session_verify_agent._MAX_ANTHROPIC_INCOMPLETE_TURN_RETRIES + 1


def _successful_sample() -> Sample:
    return Sample(
        metadata={
            "endpoint": "anthropic",
            "driver_events": anthropic_session_verify_agent._expected_driver_events(include_system=True),
            "request_count": 6,
            "tool_use_count": 3,
            "tool_result_count": 3,
            "text_turn_count": 3,
            "tool_result_string_count": 2,
            "tool_result_list_count": 1,
            "intermediate_system_used": True,
            "leaf": {"path_node_ids": [0, 1, 2, 3, 4, 7]},
            "tito_session_mismatch": [],
        }
    )


def test_generate_injects_model_and_writes_tito_metrics(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    returned_sample = _successful_sample()

    async def fake_base_generate(input):
        assert input.sample.metadata["anthropic_model"] == "/models/test"
        assert input.sample.metadata["tito_model"] == "qwen3"
        assert input.sample.metadata["anthropic_intermediate_system_expectation"] == "required"
        return GenerateFnOutput(samples=[returned_sample])

    monkeypatch.setattr(anthropic_session_verify_agent, "_base_generate", fake_base_generate)
    input_sample = Sample()
    input_value = SimpleNamespace(
        sample=input_sample,
        args=SimpleNamespace(
            hf_checkpoint="/models/test",
            tito_model="qwen3",
            anthropic_intermediate_system_expectation="required",
        ),
    )

    output = asyncio.run(anthropic_session_verify_agent.generate(input_value))

    assert output.samples == [returned_sample]
    assert input_sample.metadata == {
        "anthropic_model": "/models/test",
        "tito_model": "qwen3",
        "anthropic_intermediate_system_expectation": "required",
    }
    [metric] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert metric["driver_events"] == anthropic_session_verify_agent._expected_driver_events(include_system=True)
    assert metric["had_assistant_mismatch"] is False
    assert metric["total_mismatches"] == 0
    assert metric["hard_mismatch_count"] == 0


def test_generate_records_hard_mismatch_without_dropping_sample(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    returned_sample = _successful_sample()
    returned_sample.metadata["tito_session_mismatch"] = [
        {
            "type": "special_token_type",
            "segment_index": 4,
            "detail": "special token differs",
        }
    ]

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[returned_sample])

    monkeypatch.setattr(anthropic_session_verify_agent, "_base_generate", fake_base_generate)
    input_value = SimpleNamespace(
        sample=Sample(),
        args=SimpleNamespace(
            hf_checkpoint="/models/test",
            tito_model="qwen3",
            anthropic_intermediate_system_expectation="required",
        ),
    )

    output = asyncio.run(anthropic_session_verify_agent.generate(input_value))

    assert output.samples == [returned_sample]
    [metric] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert metric["hard_mismatch_count"] == 1
    assert metric["hard_mismatch_types"] == ["special_token_type"]
    assert metric["hard_mismatch_example"]["segment_index"] == 4


def test_generate_journals_stale_single_turn_metadata(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    sample = _successful_sample()
    sample.metadata["request_count"] = 1

    async def fake_base_generate(input):
        return GenerateFnOutput(samples=[sample])

    monkeypatch.setattr(anthropic_session_verify_agent, "_base_generate", fake_base_generate)
    input_value = SimpleNamespace(
        sample=Sample(),
        args=SimpleNamespace(
            hf_checkpoint="/models/test",
            tito_model="qwen3",
            anthropic_intermediate_system_expectation="required",
        ),
    )

    with pytest.raises(AssertionError, match="request_count=1, expected 6"):
        asyncio.run(anthropic_session_verify_agent.generate(input_value))

    sample_record, error_record = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert sample_record["hard_mismatch_count"] == 0
    assert error_record["verification_error"]["stage"] == "anthropic_session_verify_agent.generate"


@pytest.mark.parametrize(
    ("tito_model", "route_supports", "expected"),
    [
        ("qwen3", True, True),
        ("qwen3", False, False),
        ("qwen35", True, False),
        ("qwen35", False, False),
    ],
)
def test_intermediate_system_family_gate(tito_model, route_supports, expected):
    assert (
        anthropic_session_verify_agent._verify_intermediate_system(tito_model, route_supports=route_supports)
        is expected
    )


@pytest.mark.parametrize(
    ("tito_model", "route_supports", "expectation"),
    [("qwen3", False, "required"), ("qwen35", True, "required")],
)
def test_run_agent_rejects_intermediate_system_capability_drift_before_post(
    monkeypatch, tmp_path, tito_model, route_supports, expectation
):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))

    class FakeAsyncClient:
        def __init__(self, *, timeout):
            assert timeout == 180

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            assert url == "http://session/health"
            return _Response({"status": "ok", "anthropic_intermediate_system_supported": route_supports})

        async def post(self, url, *, json):
            raise AssertionError("request must not be sent after a capability mismatch")

    monkeypatch.setattr(anthropic_session_verify_agent.httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(AssertionError, match="did not match the per-model E2E contract"):
        asyncio.run(
            anthropic_session_verify_agent.run_agent(
                "http://session",
                prompt=None,
                request_kwargs={"max_tokens": 128},
                metadata={
                    "anthropic_model": "/models/test",
                    "tito_model": tito_model,
                    "anthropic_intermediate_system_expectation": expectation,
                },
            )
        )

    [record] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert record["verification_error"]["stage"] == "anthropic_session_verify_agent.run_agent"


@pytest.mark.parametrize("tito_model", ["minimax_m25", "minimax_m27"])
def test_minimax_is_rejected_before_base_generate(monkeypatch, tito_model):
    called = False

    async def fake_base_generate(input):
        nonlocal called
        called = True
        return GenerateFnOutput(samples=[])

    monkeypatch.setattr(anthropic_session_verify_agent, "_base_generate", fake_base_generate)
    input_value = SimpleNamespace(
        sample=Sample(),
        args=SimpleNamespace(hf_checkpoint="/models/test", tito_model=tito_model),
    )

    with pytest.raises(AssertionError, match="does not support tito_model"):
        asyncio.run(anthropic_session_verify_agent.generate(input_value))
    assert called is False


@pytest.mark.parametrize("tito_model", ["minimax_m25", "minimax_m27"])
def test_run_agent_rejects_minimax_before_http(tito_model):
    with pytest.raises(AssertionError, match="does not support tito_model"):
        asyncio.run(
            anthropic_session_verify_agent.run_agent(
                "http://session",
                prompt=None,
                request_kwargs={"max_tokens": 128},
                metadata={"anthropic_model": "/models/test", "tito_model": tito_model},
            )
        )
