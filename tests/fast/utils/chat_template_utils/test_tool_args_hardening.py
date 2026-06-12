"""Adversarial tool-call ``arguments`` hardening for chat-template rendering.

OpenAI wire responses carry ``function.arguments`` as a JSON *string*; Qwen-family
templates expect a mapping and iterate ``tool_call.arguments|items`` (e.g.
``qwen3.5_fixed.jinja:117``).  ``normalize_tool_arguments(..., "dict")`` decodes the
wire string back to a dict at the render boundary.

Before hardening that decode was an unguarded ``json.loads``:

- empty-string ``""`` / malformed JSON raised an uncaught ``json.JSONDecodeError``
  (the only ``try/except`` around ``apply_chat_template`` catches ``TemplateError``,
  not ``JSONDecodeError``), crashing render / data-load;
- a value that decodes to a non-dict (list / number) or an explicit ``None`` leaked
  through to the template, where ``arguments|items`` raises and the tool-format
  fallback re-renders the same broken args -> ``ValueError``.

These pure-CPU tests pin that none of those adversarial inbound values raise and that
each yields a mapping, while the default valid-dict-JSON path is unchanged.  No
tokenizer / network is needed: ``normalize_tool_arguments`` is pure-Python and the
render smoke test feeds the bundled ``qwen3.5_fixed.jinja`` to
``apply_chat_template_from_str``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import pytest

from miles.utils.chat_template_utils import (
    TEMPLATE_DIR,
    apply_chat_template_from_str,
    normalize_tool_arguments,
)

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
    """A user -> assistant(tool_call) -> tool turn carrying *arguments* verbatim."""
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


# (id, raw arguments) — every value that previously crashed or leaked a non-mapping.
# Both the wire-string forms ("[1, 2, 3]"/"42") and the native non-dict forms
# ([1, 2, 3]/42) are covered: a native non-dict ``arguments`` is just as fatal to
# ``arguments|items`` as a string that decodes to one.
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
    """``normalize_tool_arguments(..., "dict")`` must coerce adversarial args to a
    mapping instead of raising or leaking a non-dict to the template."""
    messages = _messages_with_args(arguments)
    normalized = normalize_tool_arguments(messages, "dict")
    assert isinstance(_tool_call_arguments(normalized), Mapping)


@pytest.mark.parametrize("arguments", _ADVERSARIAL_ARGS)
def test_render_does_not_raise(arguments):
    """The full HF-Jinja render (``qwen3.5_fixed.jinja`` iterates ``arguments|items``)
    must not raise on adversarial args — render is the original crash site."""
    messages = _messages_with_args(arguments)
    rendered = apply_chat_template_from_str(_QWEN35_FIXED, messages, tools=_TOOLS)
    assert isinstance(rendered, str)


def test_malformed_preserved_under_raw_arguments():
    """Undecodable strings are preserved (not silently dropped) so the call still
    renders and the raw payload remains inspectable."""
    normalized = normalize_tool_arguments(_messages_with_args("{not json}"), "dict")
    assert _tool_call_arguments(normalized) == {"_raw_arguments": "{not json}"}

    normalized_list = normalize_tool_arguments(_messages_with_args("[1, 2, 3]"), "dict")
    assert _tool_call_arguments(normalized_list) == {"_raw_arguments": "[1, 2, 3]"}


def test_empty_and_none_become_empty_mapping():
    for arguments in ("", None):
        normalized = normalize_tool_arguments(_messages_with_args(arguments), "dict")
        assert _tool_call_arguments(normalized) == {}


def test_native_non_dict_preserved_under_raw_arguments():
    """A native (already-Python) non-dict ``arguments`` — not a wire string — is
    preserved losslessly under ``_raw_arguments``, exactly like a stringified non-dict.
    The two adversarial branches stay consistent and the payload remains inspectable
    instead of being silently rewritten to an empty mapping (matches the slime
    reference ``slime/agent/auto_sample_builder/messages.py:tool_call_arguments``)."""
    for arguments in ([1, 2, 3], 42):
        normalized = normalize_tool_arguments(_messages_with_args(arguments), "dict")
        assert _tool_call_arguments(normalized) == {"_raw_arguments": arguments}


def test_default_valid_json_path_unchanged():
    """Valid dict-JSON still decodes to the exact dict (default path, no regression)."""
    normalized = normalize_tool_arguments(_messages_with_args('{"city": "London"}'), "dict")
    assert _tool_call_arguments(normalized) == {"city": "London"}


def test_dict_arguments_passthrough_unchanged():
    """An already-dict ``arguments`` is passed through untouched."""
    normalized = normalize_tool_arguments(_messages_with_args({"city": "Paris"}), "dict")
    assert _tool_call_arguments(normalized) == {"city": "Paris"}


def test_does_not_mutate_input():
    """Normalization deep-copies — the adversarial input message is never mutated."""
    messages = _messages_with_args("[1, 2, 3]")
    saved = copy.deepcopy(messages)
    normalize_tool_arguments(messages, "dict")
    assert messages == saved


def test_outbound_json_path_keeps_string_per_openai_spec():
    """The ``"json"`` (outbound/DeepSeek) direction is untouched by the hardening:
    a dict is re-serialized to a JSON string, and a non-dict wire string is left
    as-is (the OpenAI wire contract is a JSON string)."""
    normalized = normalize_tool_arguments(_messages_with_args({"city": "London"}), "json")
    assert _tool_call_arguments(normalized) == '{"city": "London"}'

    passthrough = normalize_tool_arguments(_messages_with_args('{"city": "London"}'), "json")
    assert _tool_call_arguments(passthrough) == '{"city": "London"}'
