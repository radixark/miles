"""Adversarial tool-call ``arguments`` must not crash chat-template rendering."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import pytest

from miles.utils.chat_template_utils import TEMPLATE_DIR, apply_chat_template_from_str, normalize_tool_arguments

_QWEN35_FIXED = (TEMPLATE_DIR / "qwen3.5_fixed.jinja").read_text()

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


def _messages_with_args(arguments) -> list[dict]:
    return [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "content": "sunny", "tool_call_id": "call_1", "name": "get_weather"},
    ]


_ADVERSARIAL_ARGS = [
    pytest.param("", id="empty-string"),
    pytest.param("{not json}", id="malformed-json"),
    pytest.param("[1, 2, 3]", id="decodes-to-list"),
    pytest.param("42", id="decodes-to-number"),
    pytest.param(None, id="none"),
    pytest.param([1, 2, 3], id="native-list"),
    pytest.param(42, id="native-number"),
]


def _tool_call_arguments(messages: list[dict]) -> object:
    return messages[1]["tool_calls"][0]["function"]["arguments"]


@pytest.mark.parametrize("arguments", _ADVERSARIAL_ARGS)
def test_normalize_yields_mapping_and_does_not_raise(arguments):
    messages = _messages_with_args(arguments)
    normalized = normalize_tool_arguments(messages, "dict")
    assert isinstance(_tool_call_arguments(normalized), Mapping)


@pytest.mark.parametrize("arguments", _ADVERSARIAL_ARGS)
def test_render_does_not_raise(arguments):
    messages = _messages_with_args(arguments)
    rendered = apply_chat_template_from_str(_QWEN35_FIXED, messages, tools=_TOOLS)
    assert isinstance(rendered, str)


def test_malformed_preserved_under_raw_arguments():
    for raw in ("{not json}", "[1, 2, 3]", "42"):
        normalized = normalize_tool_arguments(_messages_with_args(raw), "dict")
        assert _tool_call_arguments(normalized) == {"_raw_arguments": raw}


def test_empty_and_none_become_empty_mapping():
    for arguments in ("", None):
        normalized = normalize_tool_arguments(_messages_with_args(arguments), "dict")
        assert _tool_call_arguments(normalized) == {}


def test_native_non_dict_preserved_under_raw_arguments():
    for arguments in ([1, 2, 3], 42, 0, []):
        normalized = normalize_tool_arguments(_messages_with_args(arguments), "dict")
        assert _tool_call_arguments(normalized) == {"_raw_arguments": arguments}


def test_default_valid_json_path_unchanged():
    normalized = normalize_tool_arguments(_messages_with_args('{"city": "London"}'), "dict")
    assert _tool_call_arguments(normalized) == {"city": "London"}


def test_dict_arguments_passthrough_unchanged():
    normalized = normalize_tool_arguments(_messages_with_args({"city": "Paris"}), "dict")
    assert _tool_call_arguments(normalized) == {"city": "Paris"}


def test_does_not_mutate_input():
    messages = _messages_with_args("[1, 2, 3]")
    saved = copy.deepcopy(messages)
    normalize_tool_arguments(messages, "dict")
    assert messages == saved


def test_outbound_json_path_keeps_string_per_openai_spec():
    normalized = normalize_tool_arguments(_messages_with_args({"city": "London"}), "json")
    assert _tool_call_arguments(normalized) == '{"city": "London"}'

    passthrough = normalize_tool_arguments(_messages_with_args('{"city": "London"}'), "json")
    assert _tool_call_arguments(passthrough) == '{"city": "London"}'
