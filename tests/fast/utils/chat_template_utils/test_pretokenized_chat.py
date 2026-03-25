"""
Unit tests for the pretokenized chat completion path.

Tests that using pretokenized_token_ids + pretokenized_num_message produces
identical token IDs as the standard apply_chat_template path.

Ported from sglang test/unit/test_pretokenized_chat.py.
"""

from copy import deepcopy

import pytest

from miles.utils.chat_template_utils.autofix import try_get_fixed_chat_template
from miles.utils.chat_template_utils.template import load_hf_chat_template
from miles.utils.test_utils.chat_template_verify import assert_pretokenized_equals_standard, simulate_pretokenized_path
from miles.utils.test_utils.mock_trajectories import (
    MultiTurnTrajectory,
    MultiUserTurnThinkingTrajectory,
    SingleToolTrajectory,
    last_user_index,
)

# ---------------------------------------------------------------------------
# Load chat templates
# ---------------------------------------------------------------------------


def _load_fixed(hf_id: str) -> str:
    path = try_get_fixed_chat_template(hf_id)
    assert path is not None, f"try_get_fixed_chat_template should resolve {hf_id}"
    with open(path) as f:
        return f.read()


TEMPLATES_WITH_THINKING = {
    "qwen3_fixed": _load_fixed("Qwen/Qwen3-0.6B"),
    "qwen3.5_fixed": _load_fixed("Qwen/Qwen3.5-0.8B"),
    "glm5": load_hf_chat_template("zai-org/GLM-5"),
    "glm47_flash": load_hf_chat_template("zai-org/GLM-4.7-Flash"),
    "qwen3_thinking_2507_fixed": _load_fixed("Qwen/Qwen3-4B-Thinking-2507"),
    "qwen3_next_thinking_fixed": _load_fixed("Qwen/Qwen3-Next-80B-A3B-Thinking"),
}

ALL_TEMPLATES = {
    **TEMPLATES_WITH_THINKING,
    "qwen3_instruct_2507": load_hf_chat_template("Qwen/Qwen3-4B-Instruct-2507"),
    "qwen3_next_instruct": load_hf_chat_template("Qwen/Qwen3-Next-80B-A3B-Instruct"),
    "qwen3_coder_next": load_hf_chat_template("Qwen/Qwen3-Coder-Next"),
    "glm4": load_hf_chat_template("THUDM/glm-4-9b-chat"),
}

# Original (unfixed) HF templates referenced by negative tests
_ORIGINAL_TEMPLATES = {
    "qwen3_original": load_hf_chat_template("Qwen/Qwen3-0.6B"),
    "qwen3_thinking_2507": load_hf_chat_template("Qwen/Qwen3-4B-Thinking-2507"),
    "qwen3_next_thinking": load_hf_chat_template("Qwen/Qwen3-Next-80B-A3B-Thinking"),
}


# ===========================================================================
# Auto-generate test cases from PRETOKENIZE_POSITIONS
# ===========================================================================

from miles.utils.test_utils.chat_template_verify import (  # noqa: E402
    INTERMEDIATE_SYSTEM_CASES,
    INTERMEDIATE_SYSTEM_THINKING_CASES,
    STANDARD_CASES,
    THINKING_CASES,
)


def _to_pytest_params(cases, include_tools=True):
    """Convert (name, cls, n, tools) tuples to pytest.param list."""
    params = []
    for name, cls, n, tools in cases:
        if include_tools:
            params.append(pytest.param(cls, n, tools, id=name))
        else:
            params.append(pytest.param(cls, n, tools, id=name))
    return params


_STANDARD_CASES = _to_pytest_params(STANDARD_CASES)
_THINKING_CASES = _to_pytest_params(THINKING_CASES)
_INTERMEDIATE_SYSTEM_CASES = _to_pytest_params(INTERMEDIATE_SYSTEM_CASES)
_INTERMEDIATE_SYSTEM_THINKING_CASES = _to_pytest_params(INTERMEDIATE_SYSTEM_THINKING_CASES)

# (chat_template, trajectory_cls, pretokenize_n) — original templates that break prefix invariant
_MISMATCH_CASES = [
    pytest.param(_ORIGINAL_TEMPLATES["qwen3_original"], SingleToolTrajectory, 3, id="qwen3_original-single_tool"),
    pytest.param(_ORIGINAL_TEMPLATES["qwen3_original"], MultiTurnTrajectory, 3, id="qwen3_original-multi_turn"),
    pytest.param(
        _ORIGINAL_TEMPLATES["qwen3_thinking_2507"], SingleToolTrajectory, 3, id="qwen3_thinking_2507-single_tool"
    ),
    pytest.param(
        _ORIGINAL_TEMPLATES["qwen3_next_thinking"], SingleToolTrajectory, 3, id="qwen3_next_thinking-single_tool"
    ),
    pytest.param(
        _ORIGINAL_TEMPLATES["qwen3_next_thinking"], MultiTurnTrajectory, 3, id="qwen3_next_thinking-multi_turn"
    ),
]

# Template parametrization lists
all_template_ids = list(ALL_TEMPLATES.keys())
all_template_values = list(ALL_TEMPLATES.values())
thinking_template_ids = list(TEMPLATES_WITH_THINKING.keys())
thinking_template_values = list(TEMPLATES_WITH_THINKING.values())


# ===========================================================================
# Core tests: all templates × all trajectory/position combinations
# ===========================================================================


@pytest.mark.parametrize("chat_template", all_template_values, ids=all_template_ids)
@pytest.mark.parametrize("trajectory_cls,pretokenize_n,tools", _STANDARD_CASES)
def test_pretokenized_equals_standard(chat_template, trajectory_cls, pretokenize_n, tools):
    """Pretokenized incremental path produces same text as standard full render."""
    assert_pretokenized_equals_standard(
        chat_template=chat_template,
        messages=deepcopy(trajectory_cls.MESSAGES),
        pretokenized_num_message=pretokenize_n,
        tools=tools,
    )


# ===========================================================================
# Thinking tests: thinking-capable templates × trajectories × enable_thinking
# ===========================================================================


@pytest.mark.parametrize("chat_template", thinking_template_values, ids=thinking_template_ids)
@pytest.mark.parametrize("trajectory_cls,pretokenize_n,tools", _THINKING_CASES)
@pytest.mark.parametrize("enable_thinking", [True, False], ids=["thinking_on", "thinking_off"])
def test_pretokenized_thinking(chat_template, trajectory_cls, pretokenize_n, tools, enable_thinking):
    """Thinking-capable templates work with pretokenized path and enable_thinking flag."""
    assert_pretokenized_equals_standard(
        chat_template=chat_template,
        messages=deepcopy(trajectory_cls.MESSAGES),
        pretokenized_num_message=pretokenize_n,
        tools=tools,
        enable_thinking=enable_thinking,
    )


# ===========================================================================
# Intermediate system message tests: templates that support them
# ===========================================================================


@pytest.mark.parametrize("chat_template", all_template_values, ids=all_template_ids)
@pytest.mark.parametrize("trajectory_cls,pretokenize_n,tools", _INTERMEDIATE_SYSTEM_CASES)
def test_pretokenized_intermediate_system(chat_template, trajectory_cls, pretokenize_n, tools):
    """All templates support intermediate system messages (converted to user role in fixed templates)."""
    assert_pretokenized_equals_standard(
        chat_template=chat_template,
        messages=deepcopy(trajectory_cls.MESSAGES),
        pretokenized_num_message=pretokenize_n,
        tools=tools,
    )


@pytest.mark.parametrize("chat_template", thinking_template_values, ids=thinking_template_ids)
@pytest.mark.parametrize("trajectory_cls,pretokenize_n,tools", _INTERMEDIATE_SYSTEM_THINKING_CASES)
@pytest.mark.parametrize("enable_thinking", [True, False], ids=["thinking_on", "thinking_off"])
def test_pretokenized_intermediate_system_thinking(
    chat_template, trajectory_cls, pretokenize_n, tools, enable_thinking
):
    """Thinking templates support intermediate system messages with thinking."""
    assert_pretokenized_equals_standard(
        chat_template=chat_template,
        messages=deepcopy(trajectory_cls.MESSAGES),
        pretokenized_num_message=pretokenize_n,
        tools=tools,
        enable_thinking=enable_thinking,
    )


# ===========================================================================
# Negative tests: original (unfixed) templates fail prefix invariant
# ===========================================================================


@pytest.mark.parametrize("chat_template,trajectory_cls,pretokenize_n", _MISMATCH_CASES)
def test_original_template_prefix_mismatch(chat_template, trajectory_cls, pretokenize_n):
    """Original templates with loop.last cause prefix mismatch (our fix resolves this)."""
    with pytest.raises(ValueError, match="Prefix mismatch"):
        simulate_pretokenized_path(
            chat_template,
            deepcopy(trajectory_cls.MESSAGES),
            pretokenize_n,
            tools=trajectory_cls.TOOLS,
        )


# ===========================================================================
# Negative test: cross-user-turn thinking compression breaks prefix invariant
# ===========================================================================

# Pretokenizing BEFORE the last user turn in a multi-user-turn thinking
# trajectory fails because templates compress reasoning_content from earlier
# turns.  This is a known template limitation, not a bug in the fixed templates.
_CROSS_USER_THINKING_N = last_user_index(MultiUserTurnThinkingTrajectory.MESSAGES)


@pytest.mark.parametrize("chat_template", thinking_template_values, ids=thinking_template_ids)
@pytest.mark.parametrize("enable_thinking", [True, False], ids=["thinking_on", "thinking_off"])
def test_cross_user_turn_thinking_prefix_mismatch(chat_template, enable_thinking):
    """Thinking templates compress reasoning_content from earlier user turns, breaking prefix invariant."""
    with pytest.raises(ValueError, match="Prefix mismatch"):
        simulate_pretokenized_path(
            chat_template,
            deepcopy(MultiUserTurnThinkingTrajectory.MESSAGES),
            _CROSS_USER_THINKING_N,
            tools=MultiUserTurnThinkingTrajectory.TOOLS,
            enable_thinking=enable_thinking,
        )
