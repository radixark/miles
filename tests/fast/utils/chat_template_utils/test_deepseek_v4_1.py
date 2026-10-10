"""The DeepSeek V4.1 bridge must render exactly what sglang's ``serving_chat``
dsv41 branch renders: the ``encoding_dsv41`` encoder, the server's default
``reasoning_effort``, client-only tool payloads, and no empty system message
unless tools need a host.
"""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="stage-a-cpu", labels=[])

import json

import pytest
from sglang.srt.entrypoints.openai import chat_encoding, encoding_dsv4, encoding_dsv41
from sglang.srt.entrypoints.openai.protocol import Tool

from miles.utils.chat_template_utils import apply_chat_template, deepseek
from miles.utils.chat_template_utils.tito_tokenizer import (
    DeepSeekV4TITOTokenizer,
    DeepSeekV41TITOTokenizer,
    TITOTokenizerType,
)

_MSGS = {
    "no_system": [{"role": "user", "content": "Hello"}],
    "system": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}],
    "mid_system": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "Now answer in French."},
    ],
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


class _FakeTokenizer:
    def __init__(self, name_or_path: str):
        self.name_or_path = name_or_path

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, token):
        return hash(token) % 1000


def _tok_with_model_type(tmp_path, model_type: str) -> _FakeTokenizer:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return _FakeTokenizer(str(tmp_path))


def _server_default_effort(monkeypatch=None):
    return chat_encoding.default_dsv41_reasoning_effort_from_env("high")


def test_detect_by_config(tmp_path):
    assert deepseek.model_type(_tok_with_model_type(tmp_path, "deepseek_v41")) == "deepseek_v41"
    assert deepseek._FAMILIES["deepseek_v41"] is deepseek.V41


def test_v4_is_not_v41(tmp_path):
    assert deepseek._FAMILIES[deepseek.model_type(_tok_with_model_type(tmp_path, "deepseek_v4"))] is deepseek.V4


@pytest.mark.parametrize("scenario", list(_MSGS), ids=list(_MSGS))
@pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
def test_render_matches_server_encode(monkeypatch, scenario, thinking):
    monkeypatch.delenv("SGLANG_DSV41_REASONING_EFFORT", raising=False)
    thinking_mode = "thinking" if thinking else "chat"
    expected = encoding_dsv41.encode_messages(
        _MSGS[scenario], thinking_mode=thinking_mode, reasoning_effort=_server_default_effort()
    )
    assert deepseek.V41.render_messages(_MSGS[scenario], thinking_mode=thinking_mode) == expected


def test_render_differs_from_v4():
    assert deepseek.V41.render_messages(_MSGS["system"], thinking_mode="thinking") != deepseek.V4.render_messages(
        _MSGS["system"], thinking_mode="thinking"
    )


@pytest.mark.parametrize("scenario", list(_MSGS), ids=list(_MSGS))
def test_generation_prompt_strip_is_exact_prefix(scenario):
    without = deepseek.V41.render_messages(_MSGS[scenario], thinking_mode="thinking", add_generation_prompt=False)
    with_prompt = deepseek.V41.render_messages(_MSGS[scenario], thinking_mode="thinking")
    assert with_prompt.startswith(without)
    assert with_prompt != without


def test_reasoning_effort_env_default_and_request_override(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV41_REASONING_EFFORT", "max")
    assert deepseek.V41.render_messages(
        _MSGS["no_system"], thinking_mode="thinking"
    ) == encoding_dsv41.encode_messages(_MSGS["no_system"], thinking_mode="thinking", reasoning_effort="max")
    assert deepseek.V41.render_messages(
        _MSGS["no_system"], thinking_mode="thinking", reasoning_effort=0.3
    ) == encoding_dsv41.encode_messages(_MSGS["no_system"], thinking_mode="thinking", reasoning_effort=30)
    # an unsupported tier falls back to the server default, as serving_chat does
    assert deepseek.V41.render_messages(
        _MSGS["no_system"], thinking_mode="thinking", reasoning_effort="ultra"
    ) == encoding_dsv41.encode_messages(_MSGS["no_system"], thinking_mode="thinking", reasoning_effort="max")


def test_tools_use_client_only_payload(monkeypatch):
    monkeypatch.delenv("SGLANG_DSV41_REASONING_EFFORT", raising=False)
    payload = chat_encoding.dsv41_tool_payload(Tool.model_validate(_TOOLS[0]))
    assert payload != Tool.model_validate(_TOOLS[0]).model_dump()
    expected = encoding_dsv41.encode_messages(
        [{"role": "system", "content": "", "tools": [payload]}, *_MSGS["no_system"]],
        thinking_mode="chat",
        reasoning_effort=_server_default_effort(),
    )
    assert deepseek.V41.render_messages(_MSGS["no_system"], tools=_TOOLS, thinking_mode="chat") == expected
    assert "get_weather" in expected


def test_apply_chat_template_dispatch(tmp_path):
    tok = _tok_with_model_type(tmp_path, "deepseek_v41")
    ids = apply_chat_template(_MSGS["system"], tokenizer=tok, tokenize=True, thinking_mode="chat")
    assert ids == [ord(c) for c in deepseek.V41.render_messages(_MSGS["system"], thinking_mode="chat")]


def test_tito_binding(tmp_path):
    assert TITOTokenizerType.get_tokenizer_class(TITOTokenizerType.DEEPSEEKV41) is DeepSeekV41TITOTokenizer
    assert issubclass(DeepSeekV41TITOTokenizer, DeepSeekV4TITOTokenizer)
    assert (DeepSeekV41TITOTokenizer.reasoning_parser, DeepSeekV41TITOTokenizer.tool_call_parser) == (
        "deepseek-v41",
        "deepseekv41",
    )
    tok = _tok_with_model_type(tmp_path, "deepseek_v41")
    tito = DeepSeekV41TITOTokenizer(tok, chat_template_kwargs={"enable_thinking": False})
    assert tito.chat_template_kwargs["thinking"] is False
    assert encoding_dsv4 is not encoding_dsv41
