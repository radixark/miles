"""Tests for the DeepSeek V4 / V4.1 chat-template bridge and its dispatch through
``apply_chat_template``.

V4 and V4.1 render through deepseek-recipe, so every render must equal what
sglang's chat endpoint renders for the same request without input_ids; the one
intended difference is the V4 effort profile, which follows the checkpoint's
own encoder.  Detection uses a tiny tokenizer stub backed by a temporary
``config.json``; rendering uses the checkpoints' real tokenizers.
"""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="stage-a-cpu", labels=[])

import functools
import json

import pytest
from huggingface_hub import snapshot_download
from sglang.srt.entrypoints.openai.chat_encoding_adapters import ChatOutputHooks
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_deepseek_recipe import DeepSeekRecipeChatEncodingAdapter
from transformers import AutoTokenizer

from miles.utils.chat_template_utils import apply_chat_template, deepseek
from miles.utils.chat_template_utils.tito_tokenizer import DeepSeekV4TITOTokenizer, DeepSeekV41TITOTokenizer

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


# ---------------------------------------------------------------------------
# Detection (by config.json model_type only)
# ---------------------------------------------------------------------------


def test_detect_dsv4_by_config(tmp_path):
    assert deepseek.model_type(_tok_with_model_type(tmp_path, "deepseek_v4")) == "deepseek_v4"


def test_detect_non_dsv4(tmp_path):
    assert deepseek.model_type(_tok_with_model_type(tmp_path, "qwen3")) != "deepseek_v4"


def test_detect_dsv32_is_not_dsv4(tmp_path):
    # Cross-version exactness: a V3.2 checkpoint is not detected as V4.
    assert deepseek.model_type(_tok_with_model_type(tmp_path, "deepseek_v32")) != "deepseek_v4"


def test_detect_ignores_architectures(tmp_path):
    # Detection is by model_type ONLY: an architectures hint must not flip it.
    tok = _tok_with_model_type(tmp_path, "deepseek_v3", architectures=["DeepseekV4ForCausalLM"])
    assert deepseek.model_type(tok) != "deepseek_v4"


def test_detect_missing_config_falls_back(tmp_path):
    # No config.json -> empty model_type -> not dsv4, no exception.
    assert deepseek.model_type(_FakeTokenizer(str(tmp_path))) != "deepseek_v4"


def test_detect_invalid_config_falls_back(tmp_path):
    # Malformed JSON must fall back to HF, not raise.
    (tmp_path / "config.json").write_text("{ not valid json", encoding="utf-8")
    assert deepseek.model_type(_FakeTokenizer(str(tmp_path))) != "deepseek_v4"


def test_detect_non_object_config_falls_back(tmp_path):
    # Valid JSON that is not an object (e.g. a list) must fall back to HF, not raise.
    (tmp_path / "config.json").write_text("[]", encoding="utf-8")
    assert deepseek.model_type(_FakeTokenizer(str(tmp_path))) != "deepseek_v4"


def test_detect_empty_name_or_path():
    assert deepseek.model_type(_FakeTokenizer("")) != "deepseek_v4"


# ---------------------------------------------------------------------------
# Dispatch through apply_chat_template
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


def test_dsv32_tokenizer_not_routed_to_dsv4(tmp_path):
    # A V3.2 checkpoint routes to the V3.2 encoder: its <functions> tool block,
    # not V4's "## Tools" one.
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    out = apply_chat_template(_MSGS_BASIC, tokenizer=tok, tools=_TOOLS, tokenize=False)
    assert "<functions>" in out
    assert "### Available Tool Schemas" not in out


def test_non_deepseek_uses_hf_fallback(tmp_path):
    # A non-DeepSeek checkpoint bypasses both bridges and uses the HF chat template.
    tok = _FakeHFTokenizer(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    assert apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False) == "HF_FALLBACK_SENTINEL"


# ---------------------------------------------------------------------------
# Rendering with the real tokenizers
# ---------------------------------------------------------------------------

_CHECKPOINTS = {
    "dsv4": ("deepseek-ai/DeepSeek-V4-Flash", "60d8d70770c6776ff598c94bb586a859a38244f1"),
    "dsv41": ("deepseek-ai/DeepSeek-V4.1-Flash", "314113f8a7dadb388215396ba5fbee2c610a04d6"),
}
_TITO_CLASSES = {"dsv4": DeepSeekV4TITOTokenizer, "dsv41": DeepSeekV41TITOTokenizer}
_SPECS = list(_CHECKPOINTS)


@functools.cache
def _tokenizer(spec: str):
    repo, revision = _CHECKPOINTS[spec]
    path = snapshot_download(
        repo, revision=revision, allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json"]
    )
    return AutoTokenizer.from_pretrained(path)


@functools.cache
def _endpoint(spec: str) -> DeepSeekRecipeChatEncodingAdapter:
    hooks = ChatOutputHooks(
        thinking_requested=lambda request: True,
        tool_call_parsing_active=lambda request: False,
        history_tool_calls_count=lambda request: 0,
        tool_call_id=lambda item, count: "",
    )
    return DeepSeekRecipeChatEncodingAdapter(
        chat_encoding_spec=spec,
        tokenizer=_tokenizer(spec),
        reasoning_parser=None,
        tool_call_parser=None,
        chat_hooks=hooks,
        default_chat_template_kwargs=None,
    )


def _endpoint_ids(spec: str, messages, *, tools=None, thinking: bool, reasoning_effort=None) -> list[int]:
    """What sglang's chat endpoint renders for this request without input_ids."""
    request = ChatCompletionRequest(
        model="m",
        messages=messages,
        tools=tools,
        reasoning_effort=reasoning_effort,
        chat_template_kwargs={"thinking": thinking},
    )
    return _endpoint(spec).encode_messages(
        messages=[m.model_dump() for m in request.messages],
        request=request,
        tools=[t.model_dump() for t in request.tools] if request.tools else None,
        thinking=thinking,
        image_data=[],
    )


def _ids(spec: str, messages, **kwargs) -> list[int]:
    return apply_chat_template(messages, tokenizer=_tokenizer(spec), tokenize=True, **kwargs)


_CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
_CALL2 = {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}
_USER = {"role": "user", "content": "What's the weather in Paris?"}
_ANSWER = {"role": "assistant", "content": "It is sunny.", "reasoning_content": "Recall the forecast."}
_CALLING = {"role": "assistant", "content": "", "reasoning_content": "Need the tool.", "tool_calls": [_CALL]}
_RESULT = {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 21}'}
_FOLLOW_UP = {"role": "user", "content": "And tomorrow?"}

_SCENARIOS = {
    "user": ([_USER], None),
    "system_user": ([{"role": "system", "content": "You are helpful."}, _USER], None),
    "developer_user": ([{"role": "developer", "content": "Be terse."}, _USER], None),
    "multiturn": ([_USER, _ANSWER, _FOLLOW_UP], None),
    "mid_system": ([_USER, _ANSWER, {"role": "system", "content": "Answer in French."}, _FOLLOW_UP], None),
    "tools": ([_USER], _TOOLS),
    "tool_result": ([_USER, _CALLING, _RESULT], _TOOLS),
    "parallel_results": (
        [
            _USER,
            {**_CALLING, "tool_calls": [_CALL, _CALL2]},
            _RESULT,
            {"role": "tool", "tool_call_id": "call_2", "content": '{"temp": 25}'},
        ],
        _TOOLS,
    ),
    "tool_loop_then_user": ([_USER, _CALLING, _RESULT, _ANSWER, _FOLLOW_UP], _TOOLS),
}


@pytest.mark.parametrize("scenario", list(_SCENARIOS), ids=list(_SCENARIOS))
@pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
@pytest.mark.parametrize("spec", _SPECS)
def test_render_matches_the_chat_endpoint(spec, scenario, thinking):
    messages, tools = _SCENARIOS[scenario]
    # V4 thinking defaults to the checkpoint's "preview" high, which recipe calls low.
    endpoint_effort = "low" if spec == "dsv4" and thinking else None
    expected = _endpoint_ids(spec, messages, tools=tools, thinking=thinking, reasoning_effort=endpoint_effort)
    assert _ids(spec, messages, tools=tools, thinking=thinking) == expected


def test_v4_effort_follows_the_checkpoint_preview_profile():
    def text(**kwargs):
        return apply_chat_template(_MSGS_BASIC, tokenizer=_tokenizer("dsv4"), thinking=True, **kwargs)

    assert "Reasoning Effort" not in text()
    assert text() == text(reasoning_effort="high")
    assert text(reasoning_effort="max").startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum")
    with pytest.raises(ValueError):
        text(reasoning_effort="low")


def test_v41_integer_budget_and_env_default(monkeypatch):
    def text(**kwargs):
        return apply_chat_template(_MSGS_BASIC, tokenizer=_tokenizer("dsv41"), thinking=True, **kwargs)

    monkeypatch.delenv("SGLANG_DSV41_REASONING_EFFORT", raising=False)
    assert "Reasoning Effort: 30 (range 1-100" in text(reasoning_effort=30)
    monkeypatch.setenv("SGLANG_DSV41_REASONING_EFFORT", "50")
    assert "Reasoning Effort: 50 (range 1-100" in text()
    assert text(reasoning_effort="max") != text()


@pytest.mark.parametrize("spec", _SPECS)
def test_text_render_tokenizes_to_the_id_render(spec):
    messages, tools = _SCENARIOS["tool_loop_then_user"]
    text = apply_chat_template(messages, tokenizer=_tokenizer(spec), tools=tools, tokenize=False)
    assert _tokenizer(spec).encode(text, add_special_tokens=False) == _ids(spec, messages, tools=tools)


@pytest.mark.parametrize("spec", _SPECS)
def test_special_token_spellings_in_content_stay_text(spec):
    user_id = _tokenizer(spec).convert_tokens_to_ids("<｜User｜>")
    ids = _ids(spec, [{"role": "user", "content": "Explain <｜User｜> please."}])
    assert ids.count(user_id) == 1


@pytest.mark.parametrize("spec", _SPECS)
def test_encoder_kwargs_recipe_does_not_take_are_rejected(spec):
    with pytest.raises(TypeError):
        _ids(spec, _MSGS_BASIC, drop_thinking=False)


# ---------------------------------------------------------------------------
# TITO: appended turns tokenize as the tail of the full render
# ---------------------------------------------------------------------------

_APPENDS = {
    "tool_result_after_pending_call": ([_USER, _CALLING], [_RESULT]),
    "user_after_answer": ([_USER, _ANSWER], [_FOLLOW_UP]),
    # A pretokenized prefix may end on a tool result; the user turn then folds in on plain text.
    "user_after_tool_result": ([_USER, _CALLING, _RESULT], [_FOLLOW_UP]),
    "user_after_tool_loop": ([_USER, _CALLING, _RESULT, _ANSWER], [_FOLLOW_UP]),
    "injected_assistant": ([_USER, _ANSWER], [{"role": "assistant", "content": "injected"}]),
    "injected_assistant_then_user": ([_USER, _ANSWER], [{"role": "assistant", "content": "injected"}, _FOLLOW_UP]),
    "consecutive_assistants_then_user": (
        [_USER, _ANSWER],
        [{"role": "assistant", "content": "first injected"}, {"role": "assistant", "content": "second"}, _FOLLOW_UP],
    ),
}


@pytest.mark.parametrize("shape", list(_APPENDS), ids=list(_APPENDS))
@pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
@pytest.mark.parametrize("spec", _SPECS)
def test_appended_messages_tokenize_as_the_render_tail(spec, thinking, shape):
    history, appended = _APPENDS[shape]
    tito = _TITO_CLASSES[spec](_tokenizer(spec), chat_template_kwargs={"thinking": thinking})
    args = tito.default_template_args(_TOOLS)
    prefix = tito.apply_chat_template(history, add_generation_prompt=False, tokenize=True, template_args=args)
    full = tito.apply_chat_template(history + appended, add_generation_prompt=True, tokenize=True, template_args=args)
    delta = tito.tokenize_additional_messages(history, history + appended, template_args=args)
    decode = functools.partial(_tokenizer(spec).decode, skip_special_tokens=False)
    assert decode(prefix + delta) == decode(full)
    if history[-1]["role"] == "assistant":
        # A sampled turn ends on a special token, so nothing merges across the boundary.
        assert prefix + delta == full
