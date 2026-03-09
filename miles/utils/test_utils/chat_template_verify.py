"""Verify that a chat template satisfies the append-only invariant.

The append-only invariant means: rendering the first N messages (without
generation prompt) produces a string that is an exact prefix of rendering
all messages (with generation prompt).  This is required by sglang's
pretokenized prefix mechanism for agentic workflows.

Core functions are used by both the CLI script
(``scripts/tools/verify_chat_template.py``) and the test suite
(``tests/fast/utils/chat_template_utils/test_pretokenized_chat.py``).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from miles.utils.chat_template_utils.template import apply_chat_template_from_str, extract_tool_dicts


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

    Raises ``ValueError`` on prefix mismatch.
    """
    tool_dicts = extract_tool_dicts(tools)

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

    return full_text


def get_standard_result(
    chat_template: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    **template_kwargs,
) -> str:
    """Standard path: render all messages with generation prompt."""
    tool_dicts = extract_tool_dicts(tools)

    return apply_chat_template_from_str(
        chat_template,
        messages,
        add_generation_prompt=True,
        tools=tool_dicts,
        **template_kwargs,
    )


def assert_pretokenized_equals_standard(chat_template, messages, pretokenized_num_message, tools=None, **kwargs):
    """Assert pretokenized incremental path produces same text as standard full render."""
    standard = get_standard_result(chat_template, messages, tools=tools, **kwargs)
    pretokenized = simulate_pretokenized_path(chat_template, messages, pretokenized_num_message, tools=tools, **kwargs)
    assert pretokenized == standard, f"Pretokenized (N={pretokenized_num_message}) != standard"


# ---------------------------------------------------------------------------
# Non-raising verification API for CLI / programmatic use
# ---------------------------------------------------------------------------


@dataclass
class VerifyResult:
    """Result of a single append-only verification case."""

    case_name: str
    passed: bool
    error: str | None = None


def verify_append_only(
    chat_template: str,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    case_name: str = "",
    **template_kwargs,
) -> VerifyResult:
    """Check that the template satisfies the append-only invariant.

    Returns a ``VerifyResult`` instead of raising, making it suitable for
    batch verification in CLI scripts.
    """
    try:
        standard = get_standard_result(chat_template, deepcopy(messages), tools=tools, **template_kwargs)
        pretokenized = simulate_pretokenized_path(
            chat_template, deepcopy(messages), pretokenized_num_message, tools=tools, **template_kwargs
        )
        if pretokenized != standard:
            return VerifyResult(
                case_name=case_name, passed=False, error=f"Pretokenized (N={pretokenized_num_message}) != standard"
            )
        return VerifyResult(case_name=case_name, passed=True)
    except ValueError as e:
        return VerifyResult(case_name=case_name, passed=False, error=str(e))
    except Exception as e:
        return VerifyResult(case_name=case_name, passed=False, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Built-in test cases (shared between CLI and test suite)
# ---------------------------------------------------------------------------

from miles.utils.test_utils.mock_trajectories import (  # noqa: E402
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

# (case_name, trajectory_cls, pretokenize_n, tools)
STANDARD_CASES: list[tuple[str, type, int, list[dict] | None]] = [
    ("single_tool-N3", SingleToolTrajectory, 3, WEATHER_TOOLS),
    ("single_tool-N3-no_tools", SingleToolTrajectory, 3, None),
    ("multi_turn-N3", MultiTurnTrajectory, 3, WEATHER_TOOLS),
    ("multi_turn-N5", MultiTurnTrajectory, 5, WEATHER_TOOLS),
    ("multi_tool-N3", MultiToolSingleTurnTrajectory, 3, WEATHER_TOOLS),
    ("parallel-N3", ParallelToolsTrajectory, 3, WEATHER_TOOLS),
    ("long_chain-N3", LongChainTrajectory, 3, WEATHER_TOOLS),
    ("long_chain-N5", LongChainTrajectory, 5, WEATHER_TOOLS),
    ("long_chain-N7", LongChainTrajectory, 7, WEATHER_TOOLS),
    ("multi_user-N7", MultiUserToolChainTrajectory, 7, WEATHER_TOOLS),
    ("multi_user-N9", MultiUserToolChainTrajectory, 9, WEATHER_TOOLS),
    ("no_tool-N3", SimpleNoToolTrajectory, 3, None),
]

# (case_name, trajectory_cls, pretokenize_n) — for templates that support enable_thinking
THINKING_CASES: list[tuple[str, type, int]] = [
    ("single_tool-N3", SingleToolTrajectory, 3),
    ("thinking_single_tool-N3", SingleToolThinkingTrajectory, 3),
    ("thinking_multi_turn-N3", MultiTurnThinkingTrajectory, 3),
    ("thinking_multi_turn-N5", MultiTurnThinkingTrajectory, 5),
    ("thinking_long_chain-N3", LongChainThinkingTrajectory, 3),
    ("thinking_long_chain-N5", LongChainThinkingTrajectory, 5),
    ("thinking_long_chain-N7", LongChainThinkingTrajectory, 7),
]


def run_all_checks(
    chat_template: str,
    *,
    include_thinking: bool = False,
) -> list[VerifyResult]:
    """Run all built-in verification cases against *chat_template*.

    When *include_thinking* is True, also runs thinking-specific cases with
    ``enable_thinking=True`` and ``enable_thinking=False``.
    """
    results: list[VerifyResult] = []

    for case_name, traj_cls, n, tools in STANDARD_CASES:
        results.append(
            verify_append_only(chat_template, deepcopy(traj_cls.MESSAGES), n, tools=tools, case_name=case_name)
        )

    if include_thinking:
        for enable in (True, False):
            suffix = "thinking_on" if enable else "thinking_off"
            for case_name, traj_cls, n in THINKING_CASES:
                full_name = f"{case_name}[{suffix}]"
                results.append(
                    verify_append_only(
                        chat_template,
                        deepcopy(traj_cls.MESSAGES),
                        n,
                        tools=WEATHER_TOOLS,
                        case_name=full_name,
                        enable_thinking=enable,
                    )
                )

    return results
