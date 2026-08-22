import json

import pytest

from miles.rollout.session.anthropic import (
    AnthropicGenerationAbortedError,
    AnthropicProtocolError,
    anthropic_to_openai_request,
    openai_error_to_anthropic,
    openai_to_anthropic_response,
    render_anthropic_sse,
)


def _request(**overrides: object) -> dict[str, object]:
    request = {
        "model": "glm-4.7-flash",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "hello"}],
    }
    request.update(overrides)
    return request


def _response(**message_overrides: object) -> dict:
    message = {"role": "assistant", "content": "hello back"}
    message.update(message_overrides)
    return {
        "id": "chatcmpl-1",
        "model": "glm-4.7-flash",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def _parse_sse(body: bytes) -> list[tuple[str, dict]]:
    events = []
    for frame in body.decode().strip().split("\n\n"):
        event_line, data_line = frame.splitlines()
        events.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return events


def test_request_converts_system_sampling_and_tools() -> None:
    converted = anthropic_to_openai_request(
        _request(
            system=[{"type": "text", "text": "You are concise.", "cache_control": {"type": "ephemeral"}}],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop_sequences=["STOP"],
            stream=True,
            tools=[
                {
                    "name": "bash",
                    "description": "Run a command",
                    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
                }
            ],
            tool_choice={"type": "tool", "name": "bash"},
        )
    )

    assert converted["messages"] == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "hello"},
    ]
    assert converted["temperature"] == 0.7
    assert converted["top_p"] == 0.9
    assert converted["top_k"] == 40
    assert converted["stop"] == ["STOP"]
    assert converted["stream"] is True
    assert converted["tools"][0]["function"]["name"] == "bash"
    assert converted["tool_choice"] == {"type": "function", "function": {"name": "bash"}}


def test_request_preserves_assistant_thinking_and_tool_use() -> None:
    converted = anthropic_to_openai_request(
        _request(
            messages=[
                {"role": "user", "content": "inspect the repo"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I should list files."},
                        {"type": "text", "text": "I will inspect it."},
                        {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}},
                    ],
                },
            ]
        )
    )

    assistant = converted["messages"][1]
    assert assistant["reasoning_content"] == "I should list files."
    assert assistant["content"] == "I will inspect it."
    assert assistant["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command":"ls"}'},
        }
    ]


def test_request_preserves_text_tool_result_text_order() -> None:
    converted = anthropic_to_openai_request(
        _request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "output"},
                        {"type": "text", "text": "after"},
                    ],
                }
            ]
        )
    )

    assert converted["messages"] == [
        {"role": "user", "content": "before"},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "output"},
        {"role": "user", "content": "after"},
    ]


def test_request_converts_common_claude_code_history_shapes() -> None:
    converted = anthropic_to_openai_request(
        _request(
            system="You are concise.",
            messages=[
                {"role": "user", "content": []},
                {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            ],
            tools=[
                {
                    "name": "bash",
                    "input_schema": {"type": "object"},
                    "defer_loading": True,
                }
            ],
        )
    )

    assert converted["messages"] == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Done."},
    ]
    assert converted["tools"][0]["defer_loading"] is True
    assert converted["tool_choice"] == "auto"


def test_request_converts_tool_result_content_variants() -> None:
    converted = anthropic_to_openai_request(
        _request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_empty", "content": None},
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_text",
                            "content": [{"type": "text", "text": "output"}],
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_mixed",
                            "content": [
                                {"type": "text", "text": "screenshot"},
                                {"type": "image", "source": {"type": "base64", "data": "aW1hZ2U="}},
                            ],
                        },
                    ],
                }
            ]
        )
    )

    assert converted["messages"] == [
        {"role": "tool", "tool_call_id": "toolu_empty", "content": ""},
        {"role": "tool", "tool_call_id": "toolu_text", "content": "output"},
        {
            "role": "tool",
            "tool_call_id": "toolu_mixed",
            "content": [
                {"type": "text", "text": "screenshot"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
            ],
        },
    ]


def test_request_converts_base64_and_url_images() -> None:
    converted = anthropic_to_openai_request(
        _request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these."},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": "aW1hZ2U="},
                        },
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/image.png"},
                        },
                    ],
                }
            ]
        )
    )

    assert converted["messages"][0]["content"] == [
        {"type": "text", "text": "Compare these."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aW1hZ2U="}},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]


def test_request_joins_thinking_and_defaults_tool_input() -> None:
    converted = anthropic_to_openai_request(
        _request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "First."},
                        {"type": "thinking", "thinking": "Second."},
                        {"type": "tool_use", "id": "toolu_1", "name": "bash"},
                    ],
                }
            ]
        )
    )

    assert converted["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "First.\nSecond.",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        }
    ]


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        ({"type": "auto"}, "auto"),
        ({"type": "none"}, "none"),
        ({"type": "any"}, "required"),
    ],
)
def test_request_converts_tool_choice_variants(tool_choice: dict[str, str], expected: str) -> None:
    converted = anthropic_to_openai_request(_request(tool_choice=tool_choice))

    assert converted["tool_choice"] == expected


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        pytest.param([], "request must be an object", id="request-not-object"),
        pytest.param(_request(model=""), "model must be a non-empty string", id="empty-model"),
        pytest.param(_request(max_tokens=True), "max_tokens must be a positive integer", id="boolean-max-tokens"),
        pytest.param(_request(max_tokens=0), "max_tokens must be a positive integer", id="zero-max-tokens"),
        pytest.param(_request(messages="hello"), "messages must be an array", id="messages-not-array"),
        pytest.param(_request(messages=[None]), "messages[0] must be an object", id="message-not-object"),
        pytest.param(
            _request(messages=[{"role": "developer", "content": "hello"}]),
            "messages[0].role is invalid",
            id="invalid-role",
        ),
        pytest.param(
            _request(messages=[{"role": "user", "content": 42}]),
            "content must be a string or an array",
            id="content-not-string-or-array",
        ),
        pytest.param(
            _request(messages=[{"role": "user", "content": [None]}]),
            "content[0] must be an object",
            id="content-block-not-object",
        ),
        pytest.param(
            _request(messages=[{"role": "user", "content": [{"type": "text", "text": 42}]}]),
            "text must be a string",
            id="text-not-string",
        ),
        pytest.param(
            _request(messages=[{"role": "user", "content": [{"type": "thinking", "thinking": "secret"}]}]),
            "thinking blocks are only valid in assistant messages",
            id="thinking-in-user-message",
        ),
        pytest.param(
            _request(messages=[{"role": "assistant", "content": [{"type": "thinking", "thinking": 42}]}]),
            "thinking must be a string",
            id="thinking-not-string",
        ),
        pytest.param(
            _request(messages=[{"role": "user", "content": [{"type": "redacted_thinking", "data": "x"}]}]),
            "redacted_thinking history is not supported",
            id="redacted-thinking",
        ),
        pytest.param(
            _request(messages=[{"role": "user", "content": [{"type": "tool_use", "id": "x", "name": "bash"}]}]),
            "tool_use blocks are only valid in assistant messages",
            id="tool-use-in-user-message",
        ),
        pytest.param(
            _request(
                messages=[
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "x", "name": "bash", "input": []}],
                    }
                ]
            ),
            "tool_use.input must be an object",
            id="tool-input-not-object",
        ),
        pytest.param(
            _request(messages=[{"role": "assistant", "content": [{"type": "tool_result", "tool_use_id": "x"}]}]),
            "tool_result blocks are only valid in user messages",
            id="tool-result-in-assistant-message",
        ),
        pytest.param(
            _request(messages=[{"role": "user", "content": [{"type": "unknown"}]}]),
            "unsupported Anthropic content block type",
            id="unknown-content-block",
        ),
        pytest.param(_request(system=42), "system must be a string or an array", id="system-not-string-or-array"),
        pytest.param(
            _request(system=[{"type": "image", "source": {}}]),
            "system content blocks must be text blocks",
            id="system-not-text",
        ),
        pytest.param(_request(tools={}), "tools must be an array", id="tools-not-array"),
        pytest.param(
            _request(tools=[{"name": "web_search", "type": "web_search_20250305"}]),
            "Anthropic server-side tools are not supported",
            id="server-side-tool",
        ),
        pytest.param(
            _request(tool_choice={"type": "parallel"}),
            "unsupported tool_choice type",
            id="unknown-tool-choice",
        ),
        pytest.param(
            _request(messages=[{"role": "user", "content": [{"type": "image", "source": None}]}]),
            "image.source must be an object",
            id="image-source-not-object",
        ),
        pytest.param(
            _request(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "data": "image", "media_type": ""}}
                        ],
                    }
                ]
            ),
            "image.source.media_type must be a non-empty string",
            id="empty-image-media-type",
        ),
        pytest.param(
            _request(messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "file"}}]}]),
            "unsupported image source type",
            id="unknown-image-source",
        ),
        pytest.param(
            _request(
                messages=[{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": 42}]}]
            ),
            "tool_result.content must be a string or an array",
            id="tool-result-content-not-string-or-array",
        ),
        pytest.param(
            _request(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "x",
                                "content": [{"type": "text", "text": 42}],
                            }
                        ],
                    }
                ]
            ),
            "tool_result.content[0].text must be a string",
            id="tool-result-text-not-string",
        ),
        pytest.param(
            _request(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "x",
                                "content": [{"type": "document"}],
                            }
                        ],
                    }
                ]
            ),
            "unsupported tool_result content block type",
            id="unknown-tool-result-content-block",
        ),
    ],
)
def test_request_rejects_unrepresentable_inputs(payload: object, error: str) -> None:
    with pytest.raises(AnthropicProtocolError) as exc_info:
        anthropic_to_openai_request(payload)

    assert error in str(exc_info.value)


def test_response_converts_reasoning_text_tools_and_usage() -> None:
    converted = openai_to_anthropic_response(
        _response(
            reasoning_content="I should inspect it.",
            content="Running a command.",
            tool_calls=[
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                }
            ],
        )
    )

    assert converted["id"].startswith("msg_")
    assert converted["type"] == "message"
    assert converted["content"] == [
        {"type": "thinking", "thinking": "I should inspect it."},
        {"type": "text", "text": "Running a command."},
        {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "pwd"}},
    ]
    assert converted["stop_reason"] == "tool_use"
    assert converted["usage"] == {"input_tokens": 11, "output_tokens": 7}


def test_response_converts_null_content_tool_call() -> None:
    converted = openai_to_anthropic_response(
        _response(
            content=None,
            tool_calls=[
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                }
            ],
        )
    )

    assert converted["content"] == [{"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "pwd"}}]
    assert converted["stop_reason"] == "tool_use"


def test_response_skips_malformed_tool_calls_and_sanitizes_arguments() -> None:
    converted = openai_to_anthropic_response(
        _response(
            content=None,
            tool_calls=[
                None,
                {"id": "bad_function", "function": None},
                {"id": "bad_json", "function": {"name": "bash", "arguments": "{"}},
                {"id": "array_json", "function": {"name": "bash", "arguments": "[]"}},
            ],
        )
    )

    assert converted["content"] == [
        {"type": "tool_use", "id": "bad_json", "name": "bash", "input": {}},
        {"type": "tool_use", "id": "array_json", "name": "bash", "input": {}},
    ]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (42, [{"type": "text", "text": "42"}]),
        (
            [
                {"type": "text", "text": "kept"},
                {"type": "image_url", "image_url": {"url": "ignored"}},
                {"type": "text", "text": 42},
                None,
            ],
            [{"type": "text", "text": "kept"}],
        ),
    ],
)
def test_response_normalizes_openai_content_parts(content: object, expected: list[dict[str, str]]) -> None:
    assert openai_to_anthropic_response(_response(content=content))["content"] == expected


def test_response_maps_tool_and_length_stop_reasons() -> None:
    tool_response = _response(tool_calls=[])
    tool_response["choices"][0]["finish_reason"] = "tool_calls"
    length_response = _response()
    length_response["choices"][0]["finish_reason"] = "length"

    assert openai_to_anthropic_response(tool_response)["stop_reason"] == "tool_use"
    assert openai_to_anthropic_response(length_response)["stop_reason"] == "max_tokens"


def test_response_rejects_aborted_generation() -> None:
    response = _response()
    response["choices"][0]["finish_reason"] = "abort"

    with pytest.raises(AnthropicGenerationAbortedError, match="upstream generation was aborted"):
        openai_to_anthropic_response(response)
    with pytest.raises(AnthropicGenerationAbortedError, match="upstream generation was aborted"):
        render_anthropic_sse(response)


def test_stream_has_protocol_order_for_text_and_tool_use() -> None:
    events = _parse_sse(
        render_anthropic_sse(
            _response(
                content="Running.",
                tool_calls=[
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                    }
                ],
            )
        )
    )

    assert [event_type for event_type, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[4][1]["content_block"]["type"] == "tool_use"
    assert events[5][1]["delta"] == {"type": "input_json_delta", "partial_json": '{"command":"pwd"}'}
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"
    assert events[-2][1]["usage"] == {"output_tokens": 7}


def test_stream_sanitizes_non_finite_tool_arguments() -> None:
    events = _parse_sse(
        render_anthropic_sse(
            _response(
                content=None,
                tool_calls=[
                    {
                        "id": "toolu_nan",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"timeout":NaN}'},
                    }
                ],
            )
        )
    )

    tool_delta = next(payload["delta"] for event_type, payload in events if event_type == "content_block_delta")
    assert tool_delta == {"type": "input_json_delta", "partial_json": "{}"}


def test_stream_emits_thinking_delta() -> None:
    events = _parse_sse(render_anthropic_sse(_response(reasoning_content="Think first.", content=None)))

    assert [event_type for event_type, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1][1]["content_block"] == {"type": "thinking", "thinking": ""}
    assert events[2][1]["delta"] == {"type": "thinking_delta", "thinking": "Think first."}


def test_stream_omits_delta_for_empty_content_fallback() -> None:
    events = _parse_sse(render_anthropic_sse(_response(content="")))

    assert [event_type for event_type, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1][1]["content_block"] == {"type": "text", "text": ""}


@pytest.mark.parametrize(
    ("status_code", "expected_type"),
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (413, "request_too_large"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (503, "api_error"),
        (529, "overloaded_error"),
    ],
)
def test_error_conversion(status_code: int, expected_type: str) -> None:
    converted = openai_error_to_anthropic(status_code, {"error": {"message": "bad request"}})
    assert converted == {"type": "error", "error": {"type": expected_type, "message": "bad request"}}


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        ({"error": "string error"}, "string error"),
        ({"message": "top-level message"}, "top-level message"),
        ("not an object", "Upstream request failed with status 500"),
        ({"error": {}}, "Upstream request failed with status 500"),
    ],
)
def test_error_conversion_message_fallbacks(payload: object, expected_message: str) -> None:
    converted = openai_error_to_anthropic(500, payload)

    assert converted["error"]["message"] == expected_message
