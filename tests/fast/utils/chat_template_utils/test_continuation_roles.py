"""Each family's continuation roles match what its renderer accepts.

``FixedTemplate.roles_after_tool_call`` / ``roles_after_no_tool_call`` decide
which role the session server accepts directly after an assistant; it refuses
the rest with HTTP 400 before rendering.  The family's template or recipe still
raises on what it cannot render.  This test keeps the two in step on the real
tokenizer: a continuation the spec allows merges append-only through the TITO
path, and one it refuses makes the renderer raise.
"""

import json
from copy import deepcopy
from functools import cache

import pytest
from huggingface_hub import snapshot_download
from tests.ci.ci_register import register_cpu_ci

from miles.utils.chat_template_utils.tito_tokenizer import (
    VALID_APPEND_ROLES,
    TITOTokenizer,
    TITOTokenizerType,
    get_tito_tokenizer,
    resolve_fixed_chat_template,
)
from miles.utils.processing_utils import load_tokenizer

register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=["rollout"])

# One checkpoint per registered family class; the renderer is the class's.
_FAMILY_CHECKPOINTS = {
    TITOTokenizerType.QWEN3: "Qwen/Qwen3-0.6B",
    TITOTokenizerType.QWEN35: "Qwen/Qwen3.5-0.8B",
    TITOTokenizerType.QWEN36: "Qwen/Qwen3.6-35B-A3B",
    TITOTokenizerType.QWEN38_SMALL: "Qwen/Qwen3.8-27B",
    TITOTokenizerType.QWENNEXT: "Qwen/Qwen3-4B-Thinking-2507",
    TITOTokenizerType.GLM47: "zai-org/GLM-4.7-Flash",
    TITOTokenizerType.GLM53: "zai-org/GLM-5.3",
    TITOTokenizerType.NEMOTRON3: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    TITOTokenizerType.KIMI25: "moonshotai/Kimi-K2.5",
    TITOTokenizerType.KIMI26: "moonshotai/Kimi-K2.6",
    TITOTokenizerType.MINIMAX_M25: "MiniMaxAI/MiniMax-M2.5",
    TITOTokenizerType.MINIMAX_M27: "MiniMaxAI/MiniMax-M2.7",
    TITOTokenizerType.DEEPSEEKV32: "deepseek-ai/DeepSeek-V3.2",
    TITOTokenizerType.DEEPSEEKV4: "deepseek-ai/DeepSeek-V4-Flash",
    TITOTokenizerType.DEEPSEEKV41: "deepseek-ai/DeepSeek-V4.1-Flash",
    TITOTokenizerType.INKLING: "thinkingmachines/Inkling-Small",
}
_TOKENIZER_FILES = [
    "config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "chat_template*",
    "*.py",
    "*.model",
    "*.tiktoken",
    "tiktoken/*",
]

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]
_HISTORY = [
    {"role": "system", "content": "You are a weather assistant."},
    {"role": "user", "content": "What's the weather in Beijing?"},
]
_ASSISTANTS = {
    "tool_call": {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should call the tool.",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"location": "Beijing"})},
            }
        ],
    },
    "no_tool_call": {
        "role": "assistant",
        "content": "Beijing is a city in China.",
        "reasoning_content": "No tool is needed.",
    },
}
# A tool message after an assistant without calls has no call to answer.
_TOOL_CALL_IDS = {"tool_call": "call_0", "no_tool_call": "none"}
_APPENDED = {
    "tool": {"role": "tool", "content": '{"result": "unavailable"}'},
    "user": {"role": "user", "content": "Now check the weather in Shanghai."},
    "system": {"role": "system", "content": "Answer in a single sentence."},
    "assistant": {"role": "assistant", "content": "The earlier result was sunny."},
}


def _cases():
    for family in _FAMILY_CHECKPOINTS:
        fixed = TITOTokenizerType.get_tokenizer_class(family).FIXED_TEMPLATE
        for context, assistant in _ASSISTANTS.items():
            for role in VALID_APPEND_ROLES:
                # A role outside allowed_append_roles is refused by the role surface itself.
                if role in fixed.allowed_append_roles:
                    allowed = role in fixed.roles_after(assistant)
                    yield pytest.param(family, context, role, allowed, id=f"{family.value}-{context}-{role}")


@cache
def _tito(family: TITOTokenizerType) -> TITOTokenizer:
    path = snapshot_download(_FAMILY_CHECKPOINTS[family], allow_patterns=_TOKENIZER_FILES)
    chat_template_path, _ = resolve_fixed_chat_template(family)
    tokenizer = load_tokenizer(path, chat_template_path=chat_template_path, trust_remote_code=True)
    return get_tito_tokenizer(tokenizer, tokenizer_type=family)


def _merge_appended(tito: TITOTokenizer, context: str, role: str) -> tuple[str, str]:
    """Merge one appended message through TITO; return (merged text, canonical text)."""
    template_args = tito.default_template_args(_TOOLS)
    prefix = _HISTORY + [deepcopy(_ASSISTANTS[context])]
    appended = deepcopy(_APPENDED[role])
    if role == "tool":
        appended["tool_call_id"] = _TOOL_CALL_IDS[context]
    full = prefix + [appended]
    prefix_ids = list(
        tito.apply_chat_template(prefix, add_generation_prompt=False, tokenize=True, template_args=template_args)
    )
    # The sampled prefix ends where the model stopped, before the trailing tokens TITO re-adds.
    while prefix_ids and prefix_ids[-1] in tito.trailing_token_ids:
        prefix_ids.pop()
    merged = tito.merge_tokens(prefix, full, prefix_ids, template_args=template_args)
    expected = tito.apply_chat_template(full, add_generation_prompt=True, template_args=template_args)
    return tito.tokenizer.decode(merged), expected


def test_every_family_class_is_covered():
    registered = {TITOTokenizerType.get_tokenizer_class(family) for family in TITOTokenizerType} - {TITOTokenizer}
    covered = {TITOTokenizerType.get_tokenizer_class(family) for family in _FAMILY_CHECKPOINTS}
    assert covered == registered


@pytest.mark.parametrize("family,context,role,allowed", list(_cases()))
def test_continuation_roles_match_the_renderer(family, context, role, allowed):
    tito = _tito(family)
    if allowed:
        merged, expected = _merge_appended(tito, context, role)
        assert merged == expected
    else:
        # Exception types differ per renderer (Jinja, deepseek-recipe, the V3.2 encoder).
        with pytest.raises(Exception):  # noqa: B017
            _merge_appended(tito, context, role)
