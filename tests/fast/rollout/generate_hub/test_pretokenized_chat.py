"""
Unit tests for the pretokenized chat completion path.

Tests that using pretokenized_token_ids + pretokenized_num_message produces
identical token IDs as the standard apply_chat_template path.

Ported from sglang test/unit/test_pretokenized_chat.py.
"""

import json
from copy import deepcopy

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from miles.utils.chat_template_utils import try_get_fixed_chat_template
from miles.utils.test_utils.chat_template_loader import load_hf_chat_template
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


def _tojson(value, ensure_ascii=True, indent=None):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)


def _normalize_tool_arguments(messages: list[dict]) -> list[dict]:
    """Parse JSON-string tool_call arguments to dicts.

    Some templates (e.g. Qwen3-Coder-Next) use ``arguments|items`` which
    requires a mapping.  Others branch on ``arguments is string``.  Normalizing
    to dicts is safe for both because ``tojson(dict)`` produces the same compact
    JSON as the original string.
    """
    out = []
    for msg in messages:
        if not msg.get("tool_calls"):
            out.append(msg)
            continue
        msg = dict(msg)
        new_calls = []
        for tc in msg["tool_calls"]:
            tc = dict(tc)
            if "function" in tc:
                fn = dict(tc["function"])
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"])
                tc["function"] = fn
            new_calls.append(tc)
        msg["tool_calls"] = new_calls
        out.append(msg)
    return out


def apply_chat_template_from_str(
    chat_template: str,
    messages: list[dict],
    add_generation_prompt: bool = True,
    tools: list[dict] | None = None,
    **kwargs,
) -> str:
    """Render a Jinja2 chat template string (tokenize=False equivalent)."""
    messages = _normalize_tool_arguments(messages)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    env.filters["tojson"] = _tojson
    template = env.from_string(chat_template)

    render_kwargs = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        render_kwargs["tools"] = tools
    render_kwargs.update(kwargs)
    return template.render(**render_kwargs)


# ---------------------------------------------------------------------------
# Load chat templates from HuggingFace
# ---------------------------------------------------------------------------


def _load_fixed_template(hf_checkpoint: str) -> str:
    """Load a fixed (modified) chat template from miles/utils/chat_templates/."""
    path = try_get_fixed_chat_template(hf_checkpoint)
    assert path is not None, f"try_get_fixed_chat_template should resolve {hf_checkpoint}"
    with open(path) as f:
        return f.read()


# Fixed templates: our modified versions that satisfy the append-only invariant
QWEN3_FIXED_CHAT_TEMPLATE = _load_fixed_template("Qwen/Qwen3-0.6B")
QWEN3_THINKING_2507_FIXED_CHAT_TEMPLATE = _load_fixed_template("Qwen/Qwen3-4B-Thinking-2507")
QWEN3_NEXT_THINKING_FIXED_CHAT_TEMPLATE = _load_fixed_template("Qwen/Qwen3-Next-80B-A3B-Thinking")

# Original (unmodified) HF templates: pulled from HuggingFace, locally cached
QWEN3_ORIGINAL_CHAT_TEMPLATE = load_hf_chat_template("Qwen/Qwen3-0.6B")
QWEN35_CHAT_TEMPLATE = load_hf_chat_template("Qwen/Qwen3.5-0.8B")
GLM5_CHAT_TEMPLATE = load_hf_chat_template("zai-org/GLM-5")
QWEN3_INSTRUCT_2507_CHAT_TEMPLATE = load_hf_chat_template("Qwen/Qwen3-4B-Instruct-2507")
QWEN3_THINKING_2507_CHAT_TEMPLATE = load_hf_chat_template("Qwen/Qwen3-4B-Thinking-2507")
QWEN3_NEXT_INSTRUCT_CHAT_TEMPLATE = load_hf_chat_template("Qwen/Qwen3-Next-80B-A3B-Instruct")
QWEN3_NEXT_THINKING_CHAT_TEMPLATE = load_hf_chat_template("Qwen/Qwen3-Next-80B-A3B-Thinking")
QWEN3_CODER_NEXT_CHAT_TEMPLATE = load_hf_chat_template("Qwen/Qwen3-Coder-Next")

TEMPLATES_WITH_THINKING = {
    "qwen3_fixed": QWEN3_FIXED_CHAT_TEMPLATE,
    "qwen3.5": QWEN35_CHAT_TEMPLATE,
    "glm5": GLM5_CHAT_TEMPLATE,
    "qwen3_thinking_2507_fixed": QWEN3_THINKING_2507_FIXED_CHAT_TEMPLATE,
    "qwen3_next_thinking_fixed": QWEN3_NEXT_THINKING_FIXED_CHAT_TEMPLATE,
}

ALL_TEMPLATES = {
    **TEMPLATES_WITH_THINKING,
    "qwen3_instruct_2507": QWEN3_INSTRUCT_2507_CHAT_TEMPLATE,
    "qwen3_next_instruct": QWEN3_NEXT_INSTRUCT_CHAT_TEMPLATE,
    "qwen3_coder_next": QWEN3_CODER_NEXT_CHAT_TEMPLATE,
}


# ---------------------------------------------------------------------------
# Helper: simulate the pretokenized incremental tokenization logic
# ---------------------------------------------------------------------------


def _extract_tool_dicts(tools: list[dict] | None) -> list[dict] | None:
    """Extract function definitions from OpenAI tool format for template rendering."""
    if not tools:
        return None
    return [t["function"] for t in tools if "function" in t]


def simulate_pretokenized_path(
    chat_template: str,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    **template_kwargs,
) -> str:
    """Simulate the pretokenized incremental path at text level.

    1. Render first N messages (no generation prompt) -> prefix_text
    2. Render ALL messages (with generation prompt) -> full_text
    3. Verify prefix_text is a prefix of full_text
    4. incremental_text = full_text[len(prefix_text):]
    5. Return prefix_text (pretokenized) + incremental_text

    Since we use the same template, prefix_text == pretokenized part,
    so the result should equal full_text.
    """
    tool_dicts = _extract_tool_dicts(tools)

    prefix_text = apply_chat_template_from_str(
        chat_template,
        messages[:pretokenized_num_message],
        add_generation_prompt=False,
        tools=tool_dicts,
        **template_kwargs,
    )

    full_text = apply_chat_template_from_str(
        chat_template,
        messages,
        add_generation_prompt=True,
        tools=tool_dicts,
        **template_kwargs,
    )

    if not full_text.startswith(prefix_text):
        raise ValueError(
            f"Prefix mismatch!\n"
            f"prefix_text ({len(prefix_text)} chars):\n{repr(prefix_text[-200:])}\n\n"
            f"full_text at same position:\n{repr(full_text[:len(prefix_text)][-200:])}"
        )

    incremental_text = full_text[len(prefix_text) :]
    result = prefix_text + incremental_text
    assert result == full_text
    return full_text


def get_standard_result(
    chat_template: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    **template_kwargs,
) -> str:
    """Standard path: render all messages with generation prompt."""
    tool_dicts = _extract_tool_dicts(tools)

    return apply_chat_template_from_str(
        chat_template,
        messages,
        add_generation_prompt=True,
        tools=tool_dicts,
        **template_kwargs,
    )


def _assert_pretokenized_equals_standard(chat_template, messages, pretokenized_num_message, tools=None, **kwargs):
    standard = get_standard_result(chat_template, messages, tools=tools, **kwargs)
    pretokenized = simulate_pretokenized_path(chat_template, messages, pretokenized_num_message, tools=tools, **kwargs)
    assert pretokenized == standard, f"Pretokenized (N={pretokenized_num_message}) != standard"


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
    pytest.param(QWEN3_ORIGINAL_CHAT_TEMPLATE, SingleToolTrajectory, 3, id="qwen3_original-single_tool"),
    pytest.param(QWEN3_ORIGINAL_CHAT_TEMPLATE, MultiTurnTrajectory, 3, id="qwen3_original-multi_turn"),
    pytest.param(QWEN3_THINKING_2507_CHAT_TEMPLATE, SingleToolTrajectory, 3, id="qwen3_thinking_2507-single_tool"),
    pytest.param(QWEN3_NEXT_THINKING_CHAT_TEMPLATE, SingleToolTrajectory, 3, id="qwen3_next_thinking-single_tool"),
    pytest.param(QWEN3_NEXT_THINKING_CHAT_TEMPLATE, MultiTurnTrajectory, 3, id="qwen3_next_thinking-multi_turn"),
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
    _assert_pretokenized_equals_standard(
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
    _assert_pretokenized_equals_standard(
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
