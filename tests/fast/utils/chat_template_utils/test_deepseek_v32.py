"""Tests for the DeepSeek V3.2 chat-template bridge and its dispatch through
``apply_chat_template``.

The bridge renders via sglang's ``encoding_dsv32.encode_messages`` (a pure
string operation, no tokenizer needed), so most cases use plain message lists.
Detection and the ``tokenize=True`` dispatch use a tiny tokenizer stub backed by
a temporary ``config.json`` -- no real DeepSeek V3.2 checkpoint is required.
"""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import copy
import json

import pytest

from miles.utils.chat_template_utils import apply_chat_template, deepseek

_MSGS_BASIC = [{"role": "user", "content": "Hello"}]


class _FakeTokenizer:
    """Minimal tokenizer stub: ``name_or_path`` drives detection, ``encode`` is a
    deterministic char->id map that asserts ``add_special_tokens=False``."""

    def __init__(self, name_or_path: str):
        self.name_or_path = name_or_path

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def _tok_with_model_type(tmp_path, model_type: str) -> _FakeTokenizer:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return _FakeTokenizer(str(tmp_path))


def _reference_encode(messages, *, thinking: bool = False, drop_thinking: bool = True) -> str:
    """The canonical V3.2 rendering: a direct ``encode_messages`` call. Locks
    ``render_messages`` to this thin-bridge contract (no preprocessing of its own)."""
    from sglang.srt.entrypoints.openai import encoding_dsv32

    return encoding_dsv32.encode_messages(
        messages, thinking_mode="thinking" if thinking else "chat", drop_thinking=drop_thinking
    )


# ---------------------------------------------------------------------------
# Detection (by config.json model_type only)
# ---------------------------------------------------------------------------


def test_detect_dsv32_by_config(tmp_path):
    assert deepseek.model_type(_tok_with_model_type(tmp_path, "deepseek_v32")) == "deepseek_v32"


def test_detect_non_dsv32(tmp_path):
    assert deepseek.model_type(_tok_with_model_type(tmp_path, "qwen3")) != "deepseek_v32"


def test_detect_ignores_name(tmp_path):
    # Directory name looks like DeepSeek V3.2 but config says otherwise -> HF path.
    d = tmp_path / "deepseek-v3.2-base"
    d.mkdir()
    assert deepseek.model_type(_tok_with_model_type(d, "qwen3")) != "deepseek_v32"


def test_detect_missing_config_falls_back(tmp_path):
    # No config.json -> empty model_type -> not dsv32, no exception.
    assert deepseek.model_type(_FakeTokenizer(str(tmp_path))) != "deepseek_v32"


def test_detect_invalid_config_falls_back(tmp_path):
    # Malformed JSON must fall back to HF, not raise.
    (tmp_path / "config.json").write_text("{ not valid json", encoding="utf-8")
    assert deepseek.model_type(_FakeTokenizer(str(tmp_path))) != "deepseek_v32"


def test_detect_non_object_config_falls_back(tmp_path):
    # Valid JSON that is not an object (e.g. a list) must fall back to HF, not raise.
    (tmp_path / "config.json").write_text("[]", encoding="utf-8")
    assert deepseek.model_type(_FakeTokenizer(str(tmp_path))) != "deepseek_v32"


def test_detect_empty_name_or_path():
    assert deepseek.model_type(_FakeTokenizer("")) != "deepseek_v32"


# ---------------------------------------------------------------------------
# Rendering parity with encode_messages (full scenario matrix)
# ---------------------------------------------------------------------------

_PARITY_SCENARIOS = {
    "no_system": [{"role": "user", "content": "Hello"}],
    "system": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}],
    "tool_calls_and_result": [
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
            ],
        },
        {"role": "tool", "content": "sunny", "tool_call_id": "call_0"},
    ],
}


@pytest.mark.parametrize("scenario", list(_PARITY_SCENARIOS), ids=list(_PARITY_SCENARIOS))
@pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
def test_render_matches_direct_encode_messages(scenario, thinking):
    messages = _PARITY_SCENARIOS[scenario]
    thinking_mode = "thinking" if thinking else "chat"
    assert deepseek.V32.render_messages(messages, thinking_mode=thinking_mode) == _reference_encode(
        messages, thinking=thinking
    )


@pytest.mark.parametrize("scenario", list(_PARITY_SCENARIOS), ids=list(_PARITY_SCENARIOS))
@pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
def test_apply_chat_template_tokenize_matches_render(tmp_path, scenario, thinking):
    # tokenize=True path encodes the rendered string with add_special_tokens=False.
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    messages = _PARITY_SCENARIOS[scenario]
    thinking_mode = "thinking" if thinking else "chat"
    ids = apply_chat_template(messages, tokenizer=tok, tokenize=True, thinking_mode=thinking_mode)
    assert ids == [ord(c) for c in deepseek.V32.render_messages(messages, thinking_mode=thinking_mode)]


def test_dict_arguments_equal_string_arguments(tmp_path):
    # The dsv32 dispatch normalizes dict tool arguments to JSON strings, so dict-form
    # and string-form arguments render identically.
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    base = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"a": 1, "b": "x"}}}],
        },
        {"role": "tool", "content": "r", "tool_call_id": "c0"},
    ]
    as_string = copy.deepcopy(base)
    as_string[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"a": 1, "b": "x"}, ensure_ascii=False)
    assert apply_chat_template(base, tokenizer=tok, tokenize=False) == apply_chat_template(
        as_string, tokenizer=tok, tokenize=False
    )


def test_thinking_mode_changes_output():
    assert deepseek.V32.render_messages(_MSGS_BASIC, thinking_mode="thinking") != deepseek.V32.render_messages(
        _MSGS_BASIC, thinking_mode="chat"
    )


# ---------------------------------------------------------------------------
# Tool injection (sglang-aligned: tools go into the system <functions> block)
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]


def test_render_with_tools_injects_into_system_functions_block():
    # tools= is accepted (not rejected) and rendered into the system <functions> block.
    out = deepseek.V32.render_messages([{"role": "user", "content": "hi"}], tools=_TOOLS, thinking_mode="chat")
    assert "<functions>" in out
    assert "get_weather" in out


def test_render_with_tools_matches_manual_system_injection():
    # Passing tools= must equal pre-canonicalizing them into the system message and
    # passing tools=None — i.e. the injection is exactly the Tool-pydantic model_dump
    # that sglang's serving path applies.
    from sglang.srt.entrypoints.openai.protocol import Tool

    canonical = [Tool.model_validate(t).model_dump() for t in _TOOLS]
    msgs = [{"role": "user", "content": "weather?"}]
    expected = deepseek.V32.render_messages(
        [{"role": "system", "content": "", "tools": canonical}, *msgs], thinking_mode="chat"
    )
    assert deepseek.V32.render_messages(msgs, tools=_TOOLS, thinking_mode="chat") == expected


def test_render_with_tools_reuses_existing_system_message():
    # When a system message is already present, tools attach to it (no extra system inserted).
    msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]
    out = deepseek.V32.render_messages(msgs, tools=_TOOLS, thinking_mode="chat")
    assert "You are helpful." in out
    assert "<functions>" in out


def test_render_with_tools_does_not_mutate_input():
    msgs = [{"role": "user", "content": "hi"}]
    snapshot = copy.deepcopy(msgs)
    deepseek.V32.render_messages(msgs, tools=_TOOLS, thinking_mode="chat")
    assert msgs == snapshot


def test_apply_chat_template_with_tools_dispatches_to_bridge(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    msgs = [{"role": "user", "content": "hi"}]
    via_apply = apply_chat_template(msgs, tokenizer=tok, tools=_TOOLS, tokenize=False)
    assert via_apply == deepseek.V32.render_messages(msgs, tools=_TOOLS)


# ---------------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------------


def test_does_not_mutate_input(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    original = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"x": 2}}}],
        },
    ]
    snapshot = copy.deepcopy(original)
    apply_chat_template(original, tokenizer=tok, tokenize=False)
    assert original == snapshot


# ---------------------------------------------------------------------------
# Rejection of unknown kwargs
# ---------------------------------------------------------------------------


def test_reject_unknown_kwargs():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        deepseek.V32.render_messages(_MSGS_BASIC, some_unknown_kwarg=1)


def test_accept_none_tools_and_known_kwargs():
    deepseek.V32.render_messages(_MSGS_BASIC, tools=None, thinking_mode="thinking", drop_thinking=False)


# ---------------------------------------------------------------------------
# enable_thinking -> thinking_mode translation (miles alias for the encoder knob)
# ---------------------------------------------------------------------------


def test_enable_thinking_true_maps_to_thinking():
    assert deepseek.V32.render_messages(_MSGS_BASIC, enable_thinking=True) == deepseek.V32.render_messages(
        _MSGS_BASIC, thinking_mode="thinking"
    )


def test_enable_thinking_false_maps_to_chat():
    assert deepseek.V32.render_messages(_MSGS_BASIC, enable_thinking=False) == deepseek.V32.render_messages(
        _MSGS_BASIC, thinking_mode="chat"
    )


def test_enable_thinking_absent_defaults_to_thinking():
    # No enable_thinking and no thinking_mode -> the cfg default ("thinking").
    assert deepseek.V32.render_messages(_MSGS_BASIC) == deepseek.V32.render_messages(
        _MSGS_BASIC, thinking_mode="thinking"
    )


def test_enable_thinking_none_defaults_to_thinking():
    # Explicit None is treated as absent: falls through to the "thinking" default.
    assert deepseek.V32.render_messages(_MSGS_BASIC, enable_thinking=None) == deepseek.V32.render_messages(
        _MSGS_BASIC, thinking_mode="thinking"
    )


def test_explicit_thinking_mode_wins_over_enable_thinking():
    assert deepseek.V32.render_messages(
        _MSGS_BASIC, enable_thinking=False, thinking_mode="thinking"
    ) == deepseek.V32.render_messages(_MSGS_BASIC, thinking_mode="thinking")


def test_enable_thinking_is_consumed_not_rejected():
    # enable_thinking is translated away, so it is not rejected as an unknown kwarg.
    deepseek.V32.render_messages(_MSGS_BASIC, enable_thinking=True)


def test_build_config_does_not_mutate_input_kwargs():
    kwargs = {"enable_thinking": True}
    deepseek.V32._build_encode_config(kwargs)
    assert kwargs == {"enable_thinking": True}


# ---------------------------------------------------------------------------
# Cross-version detection exactness (the V3.2 detector must not match V4)
# ---------------------------------------------------------------------------


def test_dsv32_detector_does_not_match_dsv4(tmp_path):
    # V3.2 detection keys off model_type exactly, so a deepseek_v4 checkpoint is
    # not mistaken for V3.2 -- V4 dispatches to its own family instance.
    assert deepseek.model_type(_tok_with_model_type(tmp_path, "deepseek_v4")) != "deepseek_v32"


# ---------------------------------------------------------------------------
# Generation-prompt behavior: the encoder's auto opener honors the knob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["user", "developer"])
def test_add_generation_prompt_false_strips_the_auto_opener(role):
    for mode, opener in (("thinking", "<｜Assistant｜><think>"), ("chat", "<｜Assistant｜></think>")):
        messages = [{"role": role, "content": "Hello"}]
        with_opener = deepseek.V32.render_messages(messages, thinking_mode=mode)
        without = deepseek.V32.render_messages(messages, thinking_mode=mode, add_generation_prompt=False)
        assert with_opener == without + opener


def test_add_generation_prompt_false_strips_the_tool_tail_suffix():
    messages = _PARITY_SCENARIOS["tool_calls_and_result"]
    for mode, suffix in (("thinking", "\n\n<think>"), ("chat", "\n\n</think>")):
        with_suffix = deepseek.V32.render_messages(messages, thinking_mode=mode)
        without = deepseek.V32.render_messages(messages, thinking_mode=mode, add_generation_prompt=False)
        assert with_suffix == without + suffix


def test_add_generation_prompt_false_noop_on_partial_tool_results():
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"type": "function", "function": {"name": "g", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "content": "first", "tool_call_id": "c0"},
    ]
    for mode in ("thinking", "chat"):
        assert deepseek.V32.render_messages(
            messages, thinking_mode=mode, add_generation_prompt=False
        ) == deepseek.V32.render_messages(messages, thinking_mode=mode)


def test_add_generation_prompt_false_noop_on_assistant_tail():
    # reasoning_content is required by the thinking-mode encoder for a
    # last-round assistant message.
    msgs = _MSGS_BASIC + [{"role": "assistant", "content": "done", "reasoning_content": "r"}]
    assert deepseek.V32.render_messages(msgs, add_generation_prompt=False) == deepseek.V32.render_messages(msgs)


def test_apply_chat_template_forwards_add_generation_prompt(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    without_prompt = apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False, add_generation_prompt=False)
    assert without_prompt == deepseek.V32.render_messages(_MSGS_BASIC, add_generation_prompt=False)
    assert apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False) != without_prompt


# ---------------------------------------------------------------------------
# Dispatch integration through apply_chat_template
# ---------------------------------------------------------------------------


def test_apply_chat_template_dispatches_to_bridge(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    assert apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False) == deepseek.V32.render_messages(_MSGS_BASIC)


def test_apply_chat_template_is_generation_ready(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    out = apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False)
    assert "<｜User｜>" in out
    assert "<｜Assistant｜>" in out
