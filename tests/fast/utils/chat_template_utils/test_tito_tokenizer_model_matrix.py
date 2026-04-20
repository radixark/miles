from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pytest
from transformers import AutoTokenizer

from miles.utils.chat_template_utils import MismatchType, apply_chat_template, try_get_fixed_chat_template
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, TITOTokenizerType, get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.mock_trajectories import (
    MultiUserTurnThinkingTrajectory,
    SimpleNoToolTrajectory,
    SingleToolThinkingTrajectory,
    SingleToolTrajectory,
)

TOOL_CALL_TEST_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    # "meta-llama/Llama-3.2-1B-Instruct",  # Skipped: gated repo, requires HF_TOKEN in CI
    "zai-org/GLM-4.7-Flash",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-V3",
    "stepfun-ai/step3",
    "MiniMaxAI/MiniMax-M2",
    "MiniMaxAI/MiniMax-M2.5",
    "internlm/internlm3-8b-instruct",
    "THUDM/glm-4-9b-chat",
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2.5",
    "XiaomiMiMo/MiMo-7B-RL",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
]

# Models excluded from TITO testing due to known template incompatibilities.
# Filtered out of parametrized test cases below.
_TITO_EXCLUDED_MODELS: dict[str, str] = {
    "Qwen/Qwen3.5-0.8B": (
        "The qwen3.5 fixed template rejects non-first system messages with "
        "'System message must be at the beginning'.  TITO's synthetic bases "
        "place system first, so this exclusion may be removable — needs testing."
    ),
    "deepseek-ai/DeepSeek-V3": (
        "TITO tokenizes each tool segment independently via _tokenize_tool_segment, "
        "which causes DeepSeek-V3's template to emit extra "
        "<|tool_outputs_begin|>/<|tool_outputs_end|> wrappers that differ from "
        "full-conversation rendering."
    ),
}
_TITO_TEST_MODELS = [m for m in TOOL_CALL_TEST_MODELS if m not in _TITO_EXCLUDED_MODELS]

_ALLOWED_APPEND_ROLES = ["tool", "user", "system"]
_TOK_CACHE: dict[tuple[str, str | None], AutoTokenizer] = {}
_ASSISTANT_START_BY_MODEL: dict[str, str] = {
    "Qwen/Qwen2.5-0.5B-Instruct": "<|im_start|>assistant\n",
    "mistralai/Mistral-7B-Instruct-v0.3": "[/INST]",
    "deepseek-ai/DeepSeek-V3": "<｜Assistant｜>",
    "stepfun-ai/step3": "<|BOT|>assistant\n",
    "MiniMaxAI/MiniMax-M2": "]~b]ai\n",
    "MiniMaxAI/MiniMax-M2.5": "]~b]ai\n",
    "internlm/internlm3-8b-instruct": "<|im_start|>assistant\n",
    "THUDM/glm-4-9b-chat": "<|assistant|>",
    "moonshotai/Kimi-K2-Instruct": "<|im_assistant|>assistant<|im_middle|>",
    "moonshotai/Kimi-K2.5": "<|im_assistant|>assistant<|im_middle|>",
    "XiaomiMiMo/MiMo-7B-RL": "<|im_start|>assistant\n",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "<|im_start|>assistant\n",
}
_NO_SYSTEM_APPEND_MODELS = {
    "deepseek-ai/DeepSeek-V3",
    "stepfun-ai/step3",
    "MiniMaxAI/MiniMax-M2",
    "MiniMaxAI/MiniMax-M2.5",
}
_CONTENT_WHITESPACE_AGNOSTIC_MODELS = {
    "stepfun-ai/step3",
}


@dataclass(frozen=True)
class AppendCase:
    name: str
    old_messages: list[dict]
    appended_messages: list[dict]
    tools: list[dict] | None
    required_contents: tuple[str, ...] = ()


_APPEND_CASES = [
    AppendCase(
        name="single_tool",
        old_messages=deepcopy(SingleToolTrajectory.MESSAGES[:3]),
        appended_messages=deepcopy([SingleToolTrajectory.MESSAGES[3]]),
        tools=deepcopy(SingleToolTrajectory.TOOLS),
        required_contents=(SingleToolTrajectory.MESSAGES[3]["content"],),
    ),
    AppendCase(
        name="single_user",
        old_messages=deepcopy(MultiUserTurnThinkingTrajectory.MESSAGES[:5]),
        appended_messages=deepcopy([MultiUserTurnThinkingTrajectory.MESSAGES[5]]),
        tools=deepcopy(MultiUserTurnThinkingTrajectory.TOOLS),
        required_contents=(MultiUserTurnThinkingTrajectory.MESSAGES[5]["content"],),
    ),
    AppendCase(
        name="single_system",
        old_messages=deepcopy(SimpleNoToolTrajectory.MESSAGES),
        appended_messages=[{"role": "system", "content": "Please answer in one short sentence."}],
        tools=None,
        required_contents=("Please answer in one short sentence.",),
    ),
    AppendCase(
        name="alternating_user_tool",
        old_messages=deepcopy(SingleToolThinkingTrajectory.MESSAGES[:3]),
        appended_messages=[
            deepcopy(SingleToolThinkingTrajectory.MESSAGES[3]),
            {"role": "user", "content": "Now check Shanghai too."},
            {
                "role": "tool",
                "tool_call_id": "call_followup_1",
                "content": '{"temperature": 30, "condition": "cloudy"}',
            },
            {"role": "user", "content": "And tell me the date as well."},
        ],
        tools=deepcopy(SingleToolThinkingTrajectory.TOOLS),
        required_contents=(
            SingleToolThinkingTrajectory.MESSAGES[3]["content"],
            "Now check Shanghai too.",
            '{"temperature": 30, "condition": "cloudy"}',
            "And tell me the date as well.",
        ),
    ),
]

_ALL_PARAMS = [
    pytest.param(model_name, case, id=f"{case.name}-{model_name}")
    for model_name in _TITO_TEST_MODELS
    for case in _APPEND_CASES
    if not (case.name == "single_system" and model_name in _NO_SYSTEM_APPEND_MODELS)
]


def _resolve_tito_type(model_name: str) -> TITOTokenizerType:
    lowered = model_name.lower()
    if "qwen3" in lowered:
        return TITOTokenizerType.QWEN3
    if "glm-4.7" in lowered:
        return TITOTokenizerType.GLM47
    return TITOTokenizerType.DEFAULT


def _get_tokenizer(model_name: str) -> AutoTokenizer:
    chat_template_path = try_get_fixed_chat_template(model_name)
    cache_key = (model_name, chat_template_path)
    if cache_key not in _TOK_CACHE:
        _TOK_CACHE[cache_key] = load_tokenizer(
            model_name,
            chat_template_path=chat_template_path,
            trust_remote_code=True,
        )
    return _TOK_CACHE[cache_key]


def _get_tito(model_name: str, tokenizer: AutoTokenizer) -> TITOTokenizer:
    tokenizer_type = _resolve_tito_type(model_name)
    kwargs = {
        "tokenizer_type": tokenizer_type,
        "allowed_append_roles": _ALLOWED_APPEND_ROLES,
    }
    if tokenizer_type == TITOTokenizerType.DEFAULT:
        kwargs["assistant_start_str"] = _ASSISTANT_START_BY_MODEL[model_name]
    return get_tito_tokenizer(tokenizer, **kwargs)


def _render_ids(
    tokenizer: AutoTokenizer, messages: list[dict], tools: list[dict] | None, *, add_generation_prompt: bool
) -> list[int]:
    return apply_chat_template(
        messages,
        tokenizer=tokenizer,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
    )


def _assert_only_assistant_mismatches(tito: TITOTokenizer, expected: list[int], merged: list[int]) -> None:
    mismatches = tito.create_comparator().compare_sequences(expected, merged)
    bad = [m for m in mismatches if m.type != MismatchType.ASSISTANT_TEXT]
    assert not bad, [m.to_dict() for m in bad]


def _assert_contents_in_order(
    incremental_text: str, required_contents: tuple[str, ...], *, model_name: str, case_name: str
) -> None:
    if model_name in _CONTENT_WHITESPACE_AGNOSTIC_MODELS:
        incremental_text = "".join(incremental_text.split())
        required_contents = tuple("".join(content.split()) for content in required_contents)
    cursor = 0
    for content in required_contents:
        found = incremental_text.find(content, cursor)
        assert found >= 0, f"{model_name=} {case_name=} missing ordered content {content!r}"
        cursor = found + len(content)


def _run_case(model_name: str, case: AppendCase) -> tuple[TITOTokenizer, list[int], list[int], str]:
    tokenizer = _get_tokenizer(model_name)
    tito = _get_tito(model_name, tokenizer)
    old_messages = deepcopy(case.old_messages)
    new_messages = old_messages + deepcopy(case.appended_messages)
    try:
        expected = _render_ids(tokenizer, new_messages, case.tools, add_generation_prompt=True)
        pretokenized = _render_ids(tokenizer, old_messages, case.tools, add_generation_prompt=False)
    except Exception as exc:
        pytest.skip(f"{model_name} cannot render case {case.name}: {type(exc).__name__}: {exc}")
    merged = tito.merge_tokens(old_messages, new_messages, pretokenized, case.tools)
    incremental_text = tokenizer.decode(tito.tokenize_additional_non_assistant(old_messages, new_messages, case.tools))
    return tito, merged, expected, incremental_text


@pytest.mark.parametrize(("model_name", "case"), _ALL_PARAMS)
def test_appended_non_assistant_content_preserved(model_name: str, case: AppendCase):
    tito, merged, expected, incremental_text = _run_case(model_name, case)
    _assert_only_assistant_mismatches(tito, expected, merged)
    _assert_contents_in_order(incremental_text, case.required_contents, model_name=model_name, case_name=case.name)
