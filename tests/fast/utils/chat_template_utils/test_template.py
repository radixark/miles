"""Test that apply_chat_template aligns with SGLang's _apply_jinja_template.

The reference function ``sglang_prompt_ids`` calls
``OpenAIServingChat._process_messages`` directly — the *actual* SGLang code
path, not a re-implementation.  A lightweight ``OpenAIServingChat`` instance
is constructed via ``object.__new__`` (bypassing ``__init__``) with only the
attributes that ``_process_messages`` / ``_apply_jinja_template`` read:

- ``tokenizer_manager.tokenizer`` — the HF tokenizer under test
- ``template_manager.chat_template_name = None`` → selects the Jinja path
- ``template_manager.jinja_template_content_format = "string"`` → text-only
- ``use_dpsk_v32_encoding = False`` / ``is_gpt_oss = False``

Each test asserts that our ``apply_chat_template`` produces identical token IDs.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from transformers import AutoTokenizer

from miles.utils.chat_template_utils.template import apply_chat_template
from miles.utils.test_utils.mock_trajectories import (
    IntermediateSystemThinkingTrajectory,
    IntermediateSystemTrajectory,
    LongChainThinkingTrajectory,
    LongChainTrajectory,
    MultiToolSingleTurnTrajectory,
    MultiTurnNoToolThinkingTrajectory,
    MultiTurnNoToolTrajectory,
    MultiTurnThinkingTrajectory,
    MultiTurnTrajectory,
    MultiUserToolChainTrajectory,
    MultiUserTurnThinkingTrajectory,
    ParallelToolsTrajectory,
    RetrySystemTrajectory,
    SingleToolThinkingTrajectory,
    SingleToolTrajectory,
)

# ---------------------------------------------------------------------------
# SGLang reference: calls OpenAIServingChat._process_messages directly
# ---------------------------------------------------------------------------


def _make_serving(tokenizer) -> OpenAIServingChat:
    """Create a minimal ``OpenAIServingChat`` that can run ``_process_messages``."""
    serving = object.__new__(OpenAIServingChat)
    serving.tokenizer_manager = MagicMock()
    serving.tokenizer_manager.tokenizer = tokenizer
    serving.template_manager = MagicMock()
    serving.template_manager.chat_template_name = None
    serving.template_manager.jinja_template_content_format = "string"
    serving.use_dpsk_v32_encoding = False
    serving.is_gpt_oss = False
    serving.tool_call_parser = None
    serving.reasoning_parser = None
    return serving


def sglang_prompt_ids(
    tokenizer,
    messages: list[dict],
    tools: list[dict] | None = None,
    **kwargs,
) -> list[int]:
    """Get prompt_ids by calling SGLang's ``_process_messages`` directly."""
    request_data: dict = {"messages": copy.deepcopy(messages), "model": "test"}
    if tools:
        request_data["tools"] = copy.deepcopy(tools)
    if kwargs:
        request_data["chat_template_kwargs"] = kwargs
    request = ChatCompletionRequest(**request_data)

    serving = _make_serving(tokenizer)
    result = serving._process_messages(request, is_multimodal=False)
    return result.prompt_ids


# ---------------------------------------------------------------------------
# Tokenizer cache & fixtures
# ---------------------------------------------------------------------------

_TOK_CACHE: dict[str, AutoTokenizer] = {}


def _get_tokenizer(model_id: str) -> AutoTokenizer:
    if model_id not in _TOK_CACHE:
        _TOK_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return _TOK_CACHE[model_id]


_MODEL_IDS = [
    "Qwen/Qwen3-4B",
    "zai-org/GLM-4.7-Flash",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3-Coder-Next",
]

# Fixed chat templates — keyed by model ID, loaded from bundled .jinja files.
_TEMPLATES_DIR = Path(__file__).resolve().parents[4] / "miles" / "utils" / "chat_template_utils" / "templates"
_FIXED_CHAT_TEMPLATES: dict[str, str] = {
    "Qwen/Qwen3.5-4B": (_TEMPLATES_DIR / "qwen3.5_fixed.jinja").read_text(),
}


@pytest.fixture(params=_MODEL_IDS, ids=[m.split("/")[-1] for m in _MODEL_IDS])
def tokenizer(request) -> AutoTokenizer:
    return _get_tokenizer(request.param)


# ---------------------------------------------------------------------------
# Trajectory / kwargs definitions
# ---------------------------------------------------------------------------

_NO_INTERMEDIATE_SYSTEM_MODELS = {
    "Qwen/Qwen3.5-4B",
}

_STANDARD_CASES = [
    pytest.param(SingleToolTrajectory, {}, id="single_tool"),
    pytest.param(MultiTurnTrajectory, {}, id="multi_turn"),
    pytest.param(MultiToolSingleTurnTrajectory, {}, id="multi_tool_single_turn"),
    pytest.param(ParallelToolsTrajectory, {}, id="parallel_tools"),
    pytest.param(LongChainTrajectory, {}, id="long_chain"),
    pytest.param(MultiUserToolChainTrajectory, {}, id="multi_user_tool_chain"),
    pytest.param(MultiTurnNoToolTrajectory, {}, id="multi_turn_no_tool"),
]

# Trajectories with intermediate system messages.
_INTERMEDIATE_SYSTEM_CASES = [
    pytest.param(RetrySystemTrajectory, {}, id="retry_system"),
    pytest.param(IntermediateSystemTrajectory, {}, id="intermediate_system"),
]

_THINKING_CASES = [
    pytest.param(SingleToolThinkingTrajectory, {"enable_thinking": True}, id="single_tool_thinking_on"),
    pytest.param(SingleToolThinkingTrajectory, {"enable_thinking": False}, id="single_tool_thinking_off"),
    pytest.param(MultiTurnThinkingTrajectory, {"enable_thinking": True}, id="multi_turn_thinking_on"),
    pytest.param(LongChainThinkingTrajectory, {"enable_thinking": True}, id="long_chain_thinking_on"),
    pytest.param(MultiUserTurnThinkingTrajectory, {"enable_thinking": True}, id="multi_user_turn_thinking_on"),
    pytest.param(MultiTurnNoToolThinkingTrajectory, {"enable_thinking": True}, id="multi_turn_no_tool_thinking_on"),
    pytest.param(MultiTurnNoToolThinkingTrajectory, {"enable_thinking": False}, id="multi_turn_no_tool_thinking_off"),
]

_INTERMEDIATE_SYSTEM_THINKING_CASES = [
    pytest.param(
        IntermediateSystemThinkingTrajectory, {"enable_thinking": True}, id="intermediate_system_thinking_on"
    ),
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_aligned(tokenizer, traj_cls, kwargs):
    fixed_template = _FIXED_CHAT_TEMPLATES.get(tokenizer.name_or_path)
    extra = {"chat_template": fixed_template} if fixed_template else {}
    expected = sglang_prompt_ids(tokenizer, traj_cls.MESSAGES, traj_cls.TOOLS, **kwargs, **extra)
    actual = apply_chat_template(
        traj_cls.MESSAGES, tokenizer=tokenizer, tools=traj_cls.TOOLS, tokenize=True, **kwargs, **extra
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# Tests — parametrized over models × trajectories
# ---------------------------------------------------------------------------


class TestAlignWithSGLang:
    """apply_chat_template must produce identical prompt_ids to SGLang's pipeline."""

    @pytest.mark.parametrize("traj_cls, kwargs", _STANDARD_CASES)
    def test_standard(self, tokenizer, traj_cls, kwargs):
        _assert_aligned(tokenizer, traj_cls, kwargs)

    @pytest.mark.parametrize("traj_cls, kwargs", _INTERMEDIATE_SYSTEM_CASES)
    def test_intermediate_system(self, tokenizer, traj_cls, kwargs):
        if tokenizer.name_or_path in _NO_INTERMEDIATE_SYSTEM_MODELS:
            pytest.skip(f"{tokenizer.name_or_path} intentionally forbids intermediate system messages")
        _assert_aligned(tokenizer, traj_cls, kwargs)

    @pytest.mark.parametrize("traj_cls, kwargs", _THINKING_CASES)
    def test_thinking(self, tokenizer, traj_cls, kwargs):
        _assert_aligned(tokenizer, traj_cls, kwargs)

    @pytest.mark.parametrize("traj_cls, kwargs", _INTERMEDIATE_SYSTEM_THINKING_CASES)
    def test_intermediate_system_thinking(self, tokenizer, traj_cls, kwargs):
        if tokenizer.name_or_path in _NO_INTERMEDIATE_SYSTEM_MODELS:
            pytest.skip(f"{tokenizer.name_or_path} intentionally forbids intermediate system messages")
        _assert_aligned(tokenizer, traj_cls, kwargs)

    def test_json_string_arguments(self, tokenizer):
        """JSON-string tool_call arguments should produce same IDs as dict arguments."""
        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                    }
                ],
            },
            {"role": "tool", "content": "sunny", "tool_call_id": "call_1", "name": "get_weather"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        fixed_template = _FIXED_CHAT_TEMPLATES.get(tokenizer.name_or_path)
        extra = {"chat_template": fixed_template} if fixed_template else {}
        expected = sglang_prompt_ids(tokenizer, messages, tools, **extra)
        actual = apply_chat_template(messages, tokenizer=tokenizer, tools=tools, tokenize=True, **extra)
        assert actual == expected

    def test_no_tools(self, tokenizer):
        """Plain conversation without tools."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        fixed_template = _FIXED_CHAT_TEMPLATES.get(tokenizer.name_or_path)
        extra = {"chat_template": fixed_template} if fixed_template else {}
        expected = sglang_prompt_ids(tokenizer, messages, **extra)
        actual = apply_chat_template(messages, tokenizer=tokenizer, tokenize=True, **extra)
        assert actual == expected

    def test_does_not_mutate_input(self, tokenizer):
        messages = copy.deepcopy(SingleToolTrajectory.MESSAGES)
        tools = copy.deepcopy(SingleToolTrajectory.TOOLS)
        saved_msgs = copy.deepcopy(messages)
        saved_tools = copy.deepcopy(tools)
        apply_chat_template(messages, tokenizer=tokenizer, tools=tools, tokenize=True)
        assert messages == saved_msgs
        assert tools == saved_tools
