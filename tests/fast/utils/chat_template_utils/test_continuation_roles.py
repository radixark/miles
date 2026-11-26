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

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.fast.utils.chat_template_utils.tito_families import FAMILY_CHECKPOINTS, load_family_tito

from miles.utils.chat_template_utils.tito_tokenizer import VALID_APPEND_ROLES, TITOTokenizer, TITOTokenizerType

register_cpu_ci(est_time=50, suite="stage-b-cpu", labels=["rollout"])

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
    for family in FAMILY_CHECKPOINTS:
        fixed = TITOTokenizerType.get_tokenizer_class(family).FIXED_TEMPLATE
        for context, assistant in _ASSISTANTS.items():
            for role in VALID_APPEND_ROLES:
                # A role outside allowed_append_roles is refused by the role surface itself.
                if role in fixed.allowed_append_roles:
                    allowed = role in fixed.roles_after(assistant)
                    yield pytest.param(family, context, role, allowed, id=f"{family.value}-{context}-{role}")


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
    covered = {TITOTokenizerType.get_tokenizer_class(family) for family in FAMILY_CHECKPOINTS}
    assert covered == registered


@pytest.mark.parametrize("family,context,role,allowed", list(_cases()))
def test_continuation_roles_match_the_renderer(family, context, role, allowed):
    tito = load_family_tito(family)
    if allowed:
        merged, expected = _merge_appended(tito, context, role)
        assert merged == expected
    else:
        # Exception types differ per renderer (Jinja, deepseek-recipe, the V3.2 encoder).
        with pytest.raises(Exception):  # noqa: B017
            _merge_appended(tito, context, role)
