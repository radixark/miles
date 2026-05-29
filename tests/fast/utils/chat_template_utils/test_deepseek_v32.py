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
import inspect
import json
from pathlib import Path

import pytest

from miles.utils.chat_template_utils import apply_chat_template, deepseek_v32

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
    assert deepseek_v32.is_deepseek_v32(_tok_with_model_type(tmp_path, "deepseek_v32")) is True


def test_detect_non_dsv32(tmp_path):
    assert deepseek_v32.is_deepseek_v32(_tok_with_model_type(tmp_path, "qwen3")) is False


def test_detect_ignores_name(tmp_path):
    # Directory name looks like DeepSeek V3.2 but config says otherwise -> HF path.
    d = tmp_path / "deepseek-v3.2-base"
    d.mkdir()
    assert deepseek_v32.is_deepseek_v32(_tok_with_model_type(d, "qwen3")) is False


def test_detect_missing_config_falls_back(tmp_path):
    # No config.json -> empty model_type -> not dsv32, no exception.
    assert deepseek_v32.is_deepseek_v32(_FakeTokenizer(str(tmp_path))) is False


def test_detect_invalid_config_falls_back(tmp_path):
    # Malformed JSON must fall back to HF, not raise.
    (tmp_path / "config.json").write_text("{ not valid json", encoding="utf-8")
    assert deepseek_v32.is_deepseek_v32(_FakeTokenizer(str(tmp_path))) is False


def test_detect_non_object_config_falls_back(tmp_path):
    # Valid JSON that is not an object (e.g. a list) must fall back to HF, not raise.
    (tmp_path / "config.json").write_text("[]", encoding="utf-8")
    assert deepseek_v32.is_deepseek_v32(_FakeTokenizer(str(tmp_path))) is False


def test_detect_empty_name_or_path():
    assert deepseek_v32.is_deepseek_v32(_FakeTokenizer("")) is False


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
    assert deepseek_v32.render_messages(messages, thinking_mode=thinking_mode) == _reference_encode(
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
    assert ids == [ord(c) for c in deepseek_v32.render_messages(messages, thinking_mode=thinking_mode)]


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
    assert deepseek_v32.render_messages(_MSGS_BASIC, thinking_mode="thinking") != deepseek_v32.render_messages(
        _MSGS_BASIC, thinking_mode="chat"
    )


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
    with pytest.raises(ValueError, match="unsupported kwargs"):
        deepseek_v32.render_messages(_MSGS_BASIC, some_unknown_kwarg=1)


def test_accept_none_tools_and_known_kwargs():
    deepseek_v32.render_messages(_MSGS_BASIC, tools=None, thinking_mode="thinking", drop_thinking=False)


# ---------------------------------------------------------------------------
# DeepSeek V4 is no longer special-cased
# ---------------------------------------------------------------------------


def test_dsv4_not_special_cased(tmp_path):
    assert deepseek_v32.is_deepseek_v32(_tok_with_model_type(tmp_path, "deepseek_v4")) is False


def test_no_dsv4_references_in_package():
    import miles.utils.chat_template_utils as ctu

    pkg_dir = Path(ctu.__file__).parent
    for py in pkg_dir.glob("*.py"):
        src = py.read_text(encoding="utf-8")
        assert "deepseek_v4" not in src, f"stale dsv4 reference in {py.name}"
        assert "encoding_dsv4" not in src, f"stale dsv4 reference in {py.name}"


# ---------------------------------------------------------------------------
# Generation-prompt behavior: no knob, no suffix surgery
# ---------------------------------------------------------------------------


def test_render_has_no_add_generation_prompt_param():
    assert "add_generation_prompt" not in inspect.signature(deepseek_v32.render_messages).parameters


def test_no_generation_prompt_suffix_strip():
    src = Path(deepseek_v32.__file__).read_text(encoding="utf-8")
    assert "_GENERATION_PROMPT_SUFFIX" not in src
    assert "<｜Assistant｜>" not in src  # no hard-coded assistant-suffix surgery


def test_apply_chat_template_add_generation_prompt_is_noop(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    with_prompt = apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False, add_generation_prompt=True)
    without_prompt = apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False, add_generation_prompt=False)
    assert with_prompt == without_prompt


# ---------------------------------------------------------------------------
# Dispatch integration through apply_chat_template
# ---------------------------------------------------------------------------


def test_apply_chat_template_dispatches_to_bridge(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    assert apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False) == deepseek_v32.render_messages(_MSGS_BASIC)


def test_apply_chat_template_is_generation_ready(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    out = apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False)
    assert "<｜User｜>" in out
    assert "<｜Assistant｜>" in out
