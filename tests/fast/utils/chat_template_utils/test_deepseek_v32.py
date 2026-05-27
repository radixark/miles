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
    """The canonical V3.2 rendering: bridge-equivalent preprocessing followed by
    a direct ``encode_messages`` call.  Locks ``render_messages`` to this contract."""
    from sglang.srt.entrypoints.openai import encoding_dsv32
    from sglang.srt.parser.jinja_template_utils import process_content_for_template_format

    prepared = copy.deepcopy(messages)
    for i, msg in enumerate(prepared):
        if msg.get("content") is None:
            msg["content"] = ""
        prepared[i] = {
            **msg,
            **process_content_for_template_format(msg, "string", [], [], [], [], use_dpsk_v32_encoding=True),
        }
        msg = prepared[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tool_call in msg["tool_calls"]:
                function = tool_call.get("function") or {}
                if isinstance(function.get("arguments"), dict):
                    function["arguments"] = json.dumps(function["arguments"], ensure_ascii=False)
    return encoding_dsv32.encode_messages(
        prepared, thinking_mode="thinking" if thinking else "chat", drop_thinking=drop_thinking
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
    assert deepseek_v32.render_messages(messages, thinking=thinking) == _reference_encode(messages, thinking=thinking)


@pytest.mark.parametrize("scenario", list(_PARITY_SCENARIOS), ids=list(_PARITY_SCENARIOS))
@pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
def test_apply_chat_template_tokenize_matches_render(tmp_path, scenario, thinking):
    # tokenize=True path encodes the rendered string with add_special_tokens=False.
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    messages = _PARITY_SCENARIOS[scenario]
    ids = apply_chat_template(messages, tokenizer=tok, tokenize=True, thinking=thinking)
    assert ids == [ord(c) for c in deepseek_v32.render_messages(messages, thinking=thinking)]


def test_dict_arguments_equal_string_arguments():
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
    assert deepseek_v32.render_messages(base) == deepseek_v32.render_messages(as_string)


def test_thinking_flag_changes_output():
    assert deepseek_v32.render_messages(_MSGS_BASIC, thinking=True) != deepseek_v32.render_messages(
        _MSGS_BASIC, thinking=False
    )


# ---------------------------------------------------------------------------
# Input immutability and content normalization
# ---------------------------------------------------------------------------


def test_content_none_not_rendered_as_literal_none():
    out = deepseek_v32.render_messages([{"role": "user", "content": None}])
    assert "None" not in out


def test_does_not_mutate_input():
    original = [
        {"role": "user", "content": None},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"x": 2}}}],
        },
    ]
    snapshot = copy.deepcopy(original)
    deepseek_v32.render_messages(original)
    assert original == snapshot


# ---------------------------------------------------------------------------
# Tool injection (serving parity); rejection of unknown kwargs and multimodal
# ---------------------------------------------------------------------------

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def test_top_level_tools_injected_into_system():
    with_tools = deepseek_v32.render_messages(_MSGS_BASIC, tools=[_WEATHER_TOOL])
    without_tools = deepseek_v32.render_messages(_MSGS_BASIC)
    assert "get_weather" in with_tools
    assert "get_weather" not in without_tools


def test_top_level_tools_equal_embedded_in_system():
    # Passing tools at the top level must be identical to embedding the same
    # (canonicalized) tools on a leading system message -- i.e. the bridge injects
    # exactly the way SGLang serving does.
    canon = deepseek_v32._canonicalize_tools([_WEATHER_TOOL])
    via_top_level = deepseek_v32.render_messages([{"role": "user", "content": "q"}], tools=[_WEATHER_TOOL])
    via_embedded = deepseek_v32.render_messages(
        [{"role": "system", "content": "", "tools": canon}, {"role": "user", "content": "q"}]
    )
    assert via_top_level == via_embedded


def test_reject_unknown_kwargs():
    with pytest.raises(ValueError, match="unsupported kwargs"):
        deepseek_v32.render_messages(_MSGS_BASIC, some_unknown_kwarg=1)


def test_accept_none_tools_and_known_kwargs():
    deepseek_v32.render_messages(_MSGS_BASIC, tools=None, thinking=True, drop_thinking=False)


def test_reject_multimodal_content():
    # DeepSeek V3.2 is text-only here; media must fail loudly, not be dropped.
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://example/a.png"}}]}]
    with pytest.raises(ValueError, match="multimodal"):
        deepseek_v32.render_messages(msgs)


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
