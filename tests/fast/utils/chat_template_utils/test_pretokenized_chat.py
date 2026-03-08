"""
Unit tests for the pretokenized chat completion path.

Tests that using pretokenized_token_ids + pretokenized_num_message produces
identical token IDs as the standard apply_chat_template path.

Ported from sglang test/unit/test_pretokenized_chat.py.
"""

from copy import deepcopy

import pytest

from miles.utils.chat_template_utils.autofix import try_get_fixed_chat_template
from miles.utils.chat_template_utils.loader import load_hf_chat_template
from miles.utils.chat_template_utils.verify import assert_pretokenized_equals_standard, simulate_pretokenized_path
from miles.utils.test_utils.mock_trajectories import (
    WEATHER_TOOLS,
    LongChainThinkingTrajectory,
    LongChainTrajectory,
    MultiToolSingleTurnTrajectory,
    MultiTurnThinkingTrajectory,
    MultiTurnTrajectory,
    MultiUserToolChainTrajectory,
    ParallelToolsTrajectory,
    SimpleNoToolTrajectory,
    SingleToolThinkingTrajectory,
    SingleToolTrajectory,
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
    "qwen3.5": load_hf_chat_template("Qwen/Qwen3.5-0.8B"),
    "glm5": load_hf_chat_template("zai-org/GLM-5"),
    "qwen3_thinking_2507_fixed": _load_fixed("Qwen/Qwen3-4B-Thinking-2507"),
    "qwen3_next_thinking_fixed": _load_fixed("Qwen/Qwen3-Next-80B-A3B-Thinking"),
}

ALL_TEMPLATES = {
    **TEMPLATES_WITH_THINKING,
    "qwen3_instruct_2507": load_hf_chat_template("Qwen/Qwen3-4B-Instruct-2507"),
    "qwen3_next_instruct": load_hf_chat_template("Qwen/Qwen3-Next-80B-A3B-Instruct"),
    "qwen3_coder_next": load_hf_chat_template("Qwen/Qwen3-Coder-Next"),
}

# Original (unfixed) HF templates referenced by negative tests
_ORIGINAL_TEMPLATES = {
    "qwen3_original": load_hf_chat_template("Qwen/Qwen3-0.6B"),
    "qwen3_thinking_2507": load_hf_chat_template("Qwen/Qwen3-4B-Thinking-2507"),
    "qwen3_next_thinking": load_hf_chat_template("Qwen/Qwen3-Next-80B-A3B-Thinking"),
}


# ===========================================================================
# Test case definitions
# ===========================================================================

# (trajectory_cls, pretokenize_n, tools) — all valid pretokenize positions per trajectory
_STANDARD_CASES = [
    pytest.param(SingleToolTrajectory, 3, WEATHER_TOOLS, id="single_tool-N3"),
    pytest.param(SingleToolTrajectory, 3, None, id="single_tool-N3-no_tools"),
    pytest.param(MultiTurnTrajectory, 3, WEATHER_TOOLS, id="multi_turn-N3"),
    pytest.param(MultiTurnTrajectory, 5, WEATHER_TOOLS, id="multi_turn-N5"),
    pytest.param(MultiToolSingleTurnTrajectory, 3, WEATHER_TOOLS, id="multi_tool-N3"),
    pytest.param(ParallelToolsTrajectory, 3, WEATHER_TOOLS, id="parallel-N3"),
    pytest.param(LongChainTrajectory, 3, WEATHER_TOOLS, id="long_chain-N3"),
    pytest.param(LongChainTrajectory, 5, WEATHER_TOOLS, id="long_chain-N5"),
    pytest.param(LongChainTrajectory, 7, WEATHER_TOOLS, id="long_chain-N7"),
    pytest.param(MultiUserToolChainTrajectory, 7, WEATHER_TOOLS, id="multi_user-N7"),
    pytest.param(MultiUserToolChainTrajectory, 9, WEATHER_TOOLS, id="multi_user-N9"),
    pytest.param(SimpleNoToolTrajectory, 3, None, id="no_tool-N3"),
]

# (trajectory_cls, pretokenize_n) — both non-thinking and thinking trajectories
_THINKING_CASES = [
    pytest.param(SingleToolTrajectory, 3, id="single_tool-N3"),
    pytest.param(SingleToolThinkingTrajectory, 3, id="thinking_single_tool-N3"),
    pytest.param(MultiTurnThinkingTrajectory, 3, id="thinking_multi_turn-N3"),
    pytest.param(MultiTurnThinkingTrajectory, 5, id="thinking_multi_turn-N5"),
    pytest.param(LongChainThinkingTrajectory, 3, id="thinking_long_chain-N3"),
    pytest.param(LongChainThinkingTrajectory, 5, id="thinking_long_chain-N5"),
    pytest.param(LongChainThinkingTrajectory, 7, id="thinking_long_chain-N7"),
]

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
@pytest.mark.parametrize("trajectory_cls,pretokenize_n", _THINKING_CASES)
@pytest.mark.parametrize("enable_thinking", [True, False], ids=["thinking_on", "thinking_off"])
def test_pretokenized_thinking(chat_template, trajectory_cls, pretokenize_n, enable_thinking):
    """Thinking-capable templates work with pretokenized path and enable_thinking flag."""
    assert_pretokenized_equals_standard(
        chat_template=chat_template,
        messages=deepcopy(trajectory_cls.MESSAGES),
        pretokenized_num_message=pretokenize_n,
        tools=WEATHER_TOOLS,
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
            tools=WEATHER_TOOLS,
        )
