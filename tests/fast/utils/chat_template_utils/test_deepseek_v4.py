"""Tests for the DeepSeek V4 chat-template bridge and its dispatch through
``apply_chat_template``.

Mirrors ``test_deepseek_v32.py``: the bridge renders via sglang's
``encoding_dsv4.encode_messages`` (a pure string operation, no tokenizer
needed), so most cases use plain message lists. Detection and the
``tokenize=True`` dispatch use a tiny tokenizer stub backed by a temporary
``config.json`` -- no real DeepSeek V4 checkpoint is required. The only
behavioral delta versus V3.2 is the optional ``reasoning_effort`` kwarg.
"""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="stage-a-cpu", labels=[])

import copy
import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from miles.utils.chat_template_utils import apply_chat_template, deepseek_v4

_MSGS_BASIC = [{"role": "user", "content": "Hello"}]


class _FakeTokenizer:
    """Minimal tokenizer stub: ``name_or_path`` drives detection, ``encode`` is a
    deterministic char->id map that asserts ``add_special_tokens=False``."""

    def __init__(self, name_or_path: str):
        self.name_or_path = name_or_path

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


class _FakeHFTokenizer(_FakeTokenizer):
    """Non-DeepSeek tokenizer stub that records an ``apply_chat_template`` call,
    so we can assert the HF fallback path is taken for unrelated checkpoints."""

    def __init__(self, name_or_path: str):
        super().__init__(name_or_path)
        self.chat_template = "dummy"

    def apply_chat_template(self, messages, **kwargs):
        return "HF_FALLBACK_SENTINEL"


def _tok_with_model_type(tmp_path, model_type: str, **extra) -> _FakeTokenizer:
    config = {"model_type": model_type, **extra}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return _FakeTokenizer(str(tmp_path))


def _reference_encode(messages, *, thinking: bool = False, drop_thinking: bool = True, **extra) -> str:
    """The canonical V4 rendering: a direct ``encode_messages`` call. Locks
    ``render_messages`` to this thin-bridge contract (no preprocessing of its own)."""
    from sglang.srt.entrypoints.openai import encoding_dsv4

    return encoding_dsv4.encode_messages(
        messages, thinking_mode="thinking" if thinking else "chat", drop_thinking=drop_thinking, **extra
    )


# ---------------------------------------------------------------------------
# Detection (by config.json model_type only)
# ---------------------------------------------------------------------------


def test_detect_dsv4_by_config(tmp_path):
    assert deepseek_v4.is_deepseek_v4(_tok_with_model_type(tmp_path, "deepseek_v4")) is True


def test_detect_non_dsv4(tmp_path):
    assert deepseek_v4.is_deepseek_v4(_tok_with_model_type(tmp_path, "qwen3")) is False


def test_detect_dsv32_is_not_dsv4(tmp_path):
    # Cross-version exactness: a V3.2 checkpoint is not detected as V4.
    assert deepseek_v4.is_deepseek_v4(_tok_with_model_type(tmp_path, "deepseek_v32")) is False


def test_detect_ignores_architectures(tmp_path):
    # Detection is by model_type ONLY: an architectures hint must not flip it.
    tok = _tok_with_model_type(tmp_path, "deepseek_v3", architectures=["DeepseekV4ForCausalLM"])
    assert deepseek_v4.is_deepseek_v4(tok) is False


def test_detect_missing_config_falls_back(tmp_path):
    # No config.json -> empty model_type -> not dsv4, no exception.
    assert deepseek_v4.is_deepseek_v4(_FakeTokenizer(str(tmp_path))) is False


def test_detect_invalid_config_falls_back(tmp_path):
    # Malformed JSON must fall back to HF, not raise.
    (tmp_path / "config.json").write_text("{ not valid json", encoding="utf-8")
    assert deepseek_v4.is_deepseek_v4(_FakeTokenizer(str(tmp_path))) is False


def test_detect_non_object_config_falls_back(tmp_path):
    # Valid JSON that is not an object (e.g. a list) must fall back to HF, not raise.
    (tmp_path / "config.json").write_text("[]", encoding="utf-8")
    assert deepseek_v4.is_deepseek_v4(_FakeTokenizer(str(tmp_path))) is False


def test_detect_empty_name_or_path():
    assert deepseek_v4.is_deepseek_v4(_FakeTokenizer("")) is False


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
    assert deepseek_v4.render_messages(messages, thinking_mode=thinking_mode) == _reference_encode(
        messages, thinking=thinking
    )


@pytest.mark.parametrize("scenario", list(_PARITY_SCENARIOS), ids=list(_PARITY_SCENARIOS))
@pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
def test_apply_chat_template_tokenize_matches_render(tmp_path, scenario, thinking):
    # tokenize=True path encodes the rendered string with add_special_tokens=False.
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
    messages = _PARITY_SCENARIOS[scenario]
    thinking_mode = "thinking" if thinking else "chat"
    ids = apply_chat_template(messages, tokenizer=tok, tokenize=True, thinking_mode=thinking_mode)
    assert ids == [ord(c) for c in deepseek_v4.render_messages(messages, thinking_mode=thinking_mode)]


def test_inherited_kwargs_pass_through():
    # The V3.2 known kwargs are inherited verbatim and forwarded to encode_messages.
    messages = _PARITY_SCENARIOS["system"]
    assert deepseek_v4.render_messages(
        messages, thinking_mode="thinking", drop_thinking=False, add_default_bos_token=False
    ) == _reference_encode(messages, thinking=True, drop_thinking=False, add_default_bos_token=False)


def test_thinking_mode_changes_output():
    assert deepseek_v4.render_messages(_MSGS_BASIC, thinking_mode="thinking") != deepseek_v4.render_messages(
        _MSGS_BASIC, thinking_mode="chat"
    )


# ---------------------------------------------------------------------------
# reasoning_effort (the single V4 delta)
# ---------------------------------------------------------------------------


def test_reasoning_effort_omitted_equals_none():
    assert deepseek_v4.render_messages(_MSGS_BASIC, thinking_mode="thinking") == deepseek_v4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking", reasoning_effort=None
    )


def test_reasoning_effort_max_differs_from_none():
    # "max" + thinking emits the max-effort prefix (sglang behavior), so output differs.
    assert deepseek_v4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking", reasoning_effort="max"
    ) != deepseek_v4.render_messages(_MSGS_BASIC, thinking_mode="thinking")


def test_reasoning_effort_max_matches_direct_encode():
    assert deepseek_v4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking", reasoning_effort="max"
    ) == _reference_encode(_MSGS_BASIC, thinking=True, reasoning_effort="max")


def test_reasoning_effort_high_accepted_and_matches_direct():
    # "high" is accepted (no exception) and equals a direct call; it is currently a
    # no-op in sglang, so we do NOT assert it differs from None.
    assert deepseek_v4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking", reasoning_effort="high"
    ) == _reference_encode(_MSGS_BASIC, thinking=True, reasoning_effort="high")


def test_reasoning_effort_invalid_value_raises():
    # Value-level validation is delegated to sglang (asserts the allowed set).
    with pytest.raises(AssertionError):
        deepseek_v4.render_messages(_MSGS_BASIC, thinking_mode="thinking", reasoning_effort="ultra")


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


def test_render_with_tools_injects_tool_schemas():
    out = deepseek_v4.render_messages([{"role": "user", "content": "hi"}], tools=_TOOLS, thinking_mode="chat")
    assert "### Available Tool Schemas" in out
    assert "get_weather" in out


def test_render_with_tools_matches_manual_system_injection():
    # Passing tools= must equal pre-canonicalizing them into the system message and
    # passing tools=None -- i.e. the injection is exactly the Tool-pydantic model_dump
    # that sglang's serving path applies.
    from sglang.srt.entrypoints.openai.protocol import Tool

    canonical = [Tool.model_validate(t).model_dump() for t in _TOOLS]
    msgs = [{"role": "user", "content": "weather?"}]
    expected = deepseek_v4.render_messages(
        [{"role": "system", "content": "", "tools": canonical}, *msgs], thinking_mode="chat"
    )
    assert deepseek_v4.render_messages(msgs, tools=_TOOLS, thinking_mode="chat") == expected


def test_render_with_tools_reuses_existing_system_message():
    msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]
    out = deepseek_v4.render_messages(msgs, tools=_TOOLS, thinking_mode="chat")
    assert "You are helpful." in out
    assert "### Available Tool Schemas" in out


def test_empty_tools_list_not_injected():
    # tools=[] is falsy (like None): neither is injected, so output matches no-tools.
    msgs = [{"role": "user", "content": "hi"}]
    assert deepseek_v4.render_messages(msgs, tools=[], thinking_mode="chat") == deepseek_v4.render_messages(
        msgs, thinking_mode="chat"
    )
    assert deepseek_v4.render_messages(msgs, tools=None, thinking_mode="chat") == deepseek_v4.render_messages(
        msgs, thinking_mode="chat"
    )


def test_bare_function_dict_rejected():
    # The DeepSeek path requires a full {"type":"function","function":{...}} dict
    # (Tool.model_validate); the HF-path leniency for bare function dicts is NOT inherited.
    bare = [{"name": "get_weather", "parameters": {"type": "object", "properties": {}}}]
    with pytest.raises(ValidationError):
        deepseek_v4.render_messages([{"role": "user", "content": "hi"}], tools=bare, thinking_mode="chat")


def test_render_with_tools_does_not_mutate_input():
    msgs = [{"role": "user", "content": "hi"}]
    snapshot = copy.deepcopy(msgs)
    deepseek_v4.render_messages(msgs, tools=_TOOLS, thinking_mode="chat")
    assert msgs == snapshot


def test_apply_chat_template_with_tools_dispatches_to_bridge(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
    msgs = [{"role": "user", "content": "hi"}]
    via_apply = apply_chat_template(msgs, tokenizer=tok, tools=_TOOLS, tokenize=False)
    assert via_apply == deepseek_v4.render_messages(msgs, tools=_TOOLS)


# ---------------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------------


def test_does_not_mutate_input(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
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


def test_dict_arguments_equal_string_arguments(tmp_path):
    # The dsv4 dispatch normalizes dict tool arguments to JSON strings, so dict-form
    # and string-form arguments render identically.
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
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


# ---------------------------------------------------------------------------
# Rejection of unknown kwargs
# ---------------------------------------------------------------------------


def test_reject_unknown_kwargs():
    with pytest.raises(ValueError, match="unsupported kwargs"):
        deepseek_v4.render_messages(_MSGS_BASIC, some_unknown_kwarg=1)


def test_accept_none_tools_and_known_kwargs():
    deepseek_v4.render_messages(
        _MSGS_BASIC, tools=None, thinking_mode="thinking", drop_thinking=False, reasoning_effort=None
    )


# ---------------------------------------------------------------------------
# enable_thinking -> thinking_mode translation (miles alias for the encoder knob)
# ---------------------------------------------------------------------------


def test_enable_thinking_true_maps_to_thinking():
    assert deepseek_v4.render_messages(_MSGS_BASIC, enable_thinking=True) == deepseek_v4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking"
    )


def test_enable_thinking_false_maps_to_chat():
    assert deepseek_v4.render_messages(_MSGS_BASIC, enable_thinking=False) == deepseek_v4.render_messages(
        _MSGS_BASIC, thinking_mode="chat"
    )


def test_enable_thinking_absent_defaults_to_thinking():
    # No enable_thinking and no thinking_mode -> the cfg default ("thinking").
    assert deepseek_v4.render_messages(_MSGS_BASIC) == deepseek_v4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking"
    )


def test_enable_thinking_none_defaults_to_thinking():
    # Explicit None is treated as absent: falls through to the "thinking" default.
    assert deepseek_v4.render_messages(_MSGS_BASIC, enable_thinking=None) == deepseek_v4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking"
    )


def test_explicit_thinking_mode_wins_over_enable_thinking():
    assert deepseek_v4.render_messages(
        _MSGS_BASIC, enable_thinking=False, thinking_mode="thinking"
    ) == deepseek_v4.render_messages(_MSGS_BASIC, thinking_mode="thinking")


def test_enable_thinking_is_consumed_not_rejected():
    # enable_thinking is translated away, so it is not rejected as an unknown kwarg.
    deepseek_v4.render_messages(_MSGS_BASIC, enable_thinking=True)


def test_build_config_does_not_mutate_input_kwargs():
    kwargs = {"enable_thinking": True}
    deepseek_v4._build_deepseek_encode_config(kwargs)
    assert kwargs == {"enable_thinking": True}


# ---------------------------------------------------------------------------
# Generation-prompt behavior: no knob, no suffix surgery
# ---------------------------------------------------------------------------


def test_render_has_no_add_generation_prompt_param():
    assert "add_generation_prompt" not in inspect.signature(deepseek_v4.render_messages).parameters


def test_no_generation_prompt_suffix_strip():
    src = Path(deepseek_v4.__file__).read_text(encoding="utf-8")
    assert "_GENERATION_PROMPT_SUFFIX" not in src
    assert "<｜Assistant｜>" not in src  # no hard-coded assistant-suffix surgery


def test_apply_chat_template_add_generation_prompt_is_noop(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
    with_prompt = apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False, add_generation_prompt=True)
    without_prompt = apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False, add_generation_prompt=False)
    assert with_prompt == without_prompt


# ---------------------------------------------------------------------------
# Dispatch integration through apply_chat_template
# ---------------------------------------------------------------------------


def test_apply_chat_template_dispatches_to_bridge(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
    assert apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False) == deepseek_v4.render_messages(_MSGS_BASIC)


def test_apply_chat_template_is_generation_ready(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
    out = apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False)
    assert "<｜User｜>" in out
    assert "<｜Assistant｜>" in out


def test_dsv32_tokenizer_not_routed_to_dsv4(tmp_path):
    # A V3.2 checkpoint routes to the V3.2 bridge, not the V4 one. With tools the two
    # encoders emit different tool blocks (<functions> vs "## Tools"), making the
    # routing observable (a bare message renders identically under both encoders).
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    out = apply_chat_template(_MSGS_BASIC, tokenizer=tok, tools=_TOOLS, tokenize=False)
    assert "<functions>" in out  # V3.2 tool block
    assert "### Available Tool Schemas" not in out  # not the V4 tool block


def test_non_deepseek_uses_hf_fallback(tmp_path):
    # A non-DeepSeek checkpoint bypasses both bridges and uses the HF chat template.
    tok = _FakeHFTokenizer(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    assert apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False) == "HF_FALLBACK_SENTINEL"
