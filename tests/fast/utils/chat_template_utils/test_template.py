"""Test that apply_chat_template aligns with SGLang's _process_messages prompt_ids.

The reference function ``sglang_prompt_ids`` replicates the exact preprocessing
from ``OpenAIServingChat._apply_jinja_template`` (serving_chat.py):
1. Parse messages through SGLang's Pydantic models.
2. Set ``content = ""`` for all None-content messages.
3. Parse JSON-string tool_call arguments to dicts via orjson.
4. Canonicalize tools via ``protocol.Tool`` → ``function.model_dump()``.
5. Call ``tokenizer.apply_chat_template`` with fallback to wrapped tool format.

Each test asserts that our ``apply_chat_template`` produces identical token IDs.
"""

from __future__ import annotations

import copy
from pathlib import Path

import orjson
import pytest
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageGenericParam,
    ChatCompletionMessageUserParam,
    Tool,
)
from transformers import AutoTokenizer

from miles.utils.chat_template_utils.template import apply_chat_template
from miles.utils.test_utils.mock_trajectories import (
    IntermediateSystemThinkingTrajectory,
    IntermediateSystemTrajectory,
    LongChainThinkingTrajectory,
    LongChainTrajectory,
    MultiToolSingleTurnTrajectory,
    MultiTurnThinkingTrajectory,
    MultiTurnTrajectory,
    MultiUserToolChainTrajectory,
    MultiUserTurnThinkingTrajectory,
    ParallelToolsTrajectory,
    RetrySystemTrajectory,
    SimpleNoToolTrajectory,
    SingleToolThinkingTrajectory,
    SingleToolTrajectory,
)

# ---------------------------------------------------------------------------
# SGLang reference implementation
# ---------------------------------------------------------------------------


def _parse_message(raw: dict):
    """Parse a raw message dict through SGLang's Pydantic model (same as FastAPI)."""
    if raw.get("role") == "user":
        return ChatCompletionMessageUserParam(**raw)
    return ChatCompletionMessageGenericParam(**raw)


def sglang_prompt_ids(
    tokenizer,
    messages: list[dict],
    tools: list[dict] | None = None,
    **kwargs,
) -> list[int]:
    """Replicate SGLang's ``_apply_jinja_template`` preprocessing → prompt_ids."""
    sglang_msgs = []
    for raw in copy.deepcopy(messages):
        msg = _parse_message(raw)
        if msg.content is None:
            msg.content = ""
        d = msg.model_dump()
        if d["role"] == "assistant" and isinstance(d.get("tool_calls"), list):
            for item in d["tool_calls"]:
                if "function" in item and isinstance(item["function"].get("arguments"), str):
                    item["function"]["arguments"] = orjson.loads(item["function"]["arguments"])
        sglang_msgs.append(d)

    tool_defs = [Tool(**t).function.model_dump() for t in tools] if tools else None

    try:
        return tokenizer.apply_chat_template(
            sglang_msgs, tokenize=True, add_generation_prompt=True, tools=tool_defs, return_dict=False, **kwargs
        )
    except Exception:
        tool_defs = [{"function": t} for t in tool_defs] if tool_defs else None
        return tokenizer.apply_chat_template(
            sglang_msgs, tokenize=True, add_generation_prompt=True, tools=tool_defs, return_dict=False, **kwargs
        )


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

_STANDARD_CASES = [
    pytest.param(SingleToolTrajectory, {}, id="single_tool"),
    pytest.param(MultiTurnTrajectory, {}, id="multi_turn"),
    pytest.param(MultiToolSingleTurnTrajectory, {}, id="multi_tool_single_turn"),
    pytest.param(ParallelToolsTrajectory, {}, id="parallel_tools"),
    pytest.param(LongChainTrajectory, {}, id="long_chain"),
    pytest.param(MultiUserToolChainTrajectory, {}, id="multi_user_tool_chain"),
    pytest.param(SimpleNoToolTrajectory, {}, id="simple_no_tool"),
]

# Trajectories with intermediate system messages (Qwen3.5 uses fixed template).
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
        _assert_aligned(tokenizer, traj_cls, kwargs)

    @pytest.mark.parametrize("traj_cls, kwargs", _THINKING_CASES)
    def test_thinking(self, tokenizer, traj_cls, kwargs):
        _assert_aligned(tokenizer, traj_cls, kwargs)

    @pytest.mark.parametrize("traj_cls, kwargs", _INTERMEDIATE_SYSTEM_THINKING_CASES)
    def test_intermediate_system_thinking(self, tokenizer, traj_cls, kwargs):
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
