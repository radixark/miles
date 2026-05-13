"""Multi-turn trajectory definitions for testing.

Each trajectory class defines a complete multi-turn conversation with tool calls.
Used by:
- tests/fast/rollout/generate_hub/test_pretokenized_chat.py (chat template verification)
- tests/fast/router/test_session_pretokenized_e2e.py (session proxy e2e)

Class attributes consumed by chat_template_verify:

- ``APPEND_ROLES: frozenset[str]`` — non-assistant roles that appear *after*
  the first assistant message (``tool`` / ``user`` / ``system``).  These are
  the roles the session must allow to be appended on top of an assistant-
  stopped prefix; drives ``--tito-allowed-append-roles`` filtering.
- ``IS_THINKING: bool`` — ``True`` iff at least one assistant message carries
  ``reasoning_content``.  Drives ``--thinking`` filtering and whether the
  ``enable_thinking`` chat-template kwarg is passed.

Both are declared explicitly on each class so readers can see a trajectory's
verify-layer classification without having to execute the module.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from miles.utils.test_utils.mock_sglang_server import ProcessFn, ProcessResult

# ---------------------------------------------------------------------------
# Shared tool definitions
# ---------------------------------------------------------------------------

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

DATE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_date",
        "description": "Get the current date and time for a timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name (e.g. Asia/Shanghai, UTC)",
                },
            },
            "required": ["timezone"],
        },
    },
}

ALL_TOOLS = WEATHER_TOOLS + [DATE_TOOL]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One turn in a multi-turn trajectory."""

    request_messages: list[dict[str, Any]]
    assistant_message: dict[str, Any]
    response_text: str = ""  # raw text returned by process_fn (computed by build_trajectory)


@dataclass
class Trajectory:
    """A fully resolved multi-turn trajectory ready for testing."""

    tools: list[dict[str, Any]] | None
    turns: list[Turn]
    full_messages: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def last_user_index(messages: list[dict[str, Any]]) -> int:
    """Return the index of the last user message in *messages*."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            return i
    raise ValueError("No user message found")


# ---------------------------------------------------------------------------
# Trajectory classes (by scenario)
# ---------------------------------------------------------------------------


class SingleToolTrajectory:
    """sys, user, assistant(tool_call), tool — 1 turn"""

    TOOLS = WEATHER_TOOLS
    PRETOKENIZE_POSITIONS = [3]
    APPEND_ROLES = frozenset({"tool"})
    IS_THINKING = False
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Beijing?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing", "unit": "celsius"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25, "condition": "sunny"}',
            "tool_call_id": "call_1",
        },
    ]


class MultiTurnTrajectory:
    """sys, user, ass(tool), tool, ass(tool), tool — 2 turns"""

    TOOLS = WEATHER_TOOLS
    PRETOKENIZE_POSITIONS = [3, 5]
    APPEND_ROLES = frozenset({"tool"})
    IS_THINKING = False
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Beijing and Shanghai?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25, "condition": "sunny"}',
            "tool_call_id": "call_1",
        },
        {
            "role": "assistant",
            "content": "Beijing is 25C. Let me check Shanghai.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Shanghai"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 30, "condition": "cloudy"}',
            "tool_call_id": "call_2",
        },
    ]


class MultiToolSingleTurnTrajectory:
    """sys, user, assistant(2 tool_calls: weather+date), tool, tool — 1 turn"""

    TOOLS = ALL_TOOLS
    PRETOKENIZE_POSITIONS = [3]
    APPEND_ROLES = frozenset({"tool"})
    IS_THINKING = False
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Beijing and what date is it?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_date",
                        "arguments": {"timezone": "Asia/Shanghai"},
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25}',
            "tool_call_id": "call_1",
        },
        {
            "role": "tool",
            "content": '{"date": "2025-03-15", "time": "14:30:00"}',
            "tool_call_id": "call_2",
        },
    ]


class ParallelToolsTrajectory:
    """sys, user, assistant(3 parallel tool_calls), tool, tool, tool — 1 turn"""

    TOOLS = WEATHER_TOOLS
    PRETOKENIZE_POSITIONS = [3]
    APPEND_ROLES = frozenset({"tool"})
    IS_THINKING = False
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Compare weather in Beijing, Shanghai, and Guangzhou"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Shanghai"},
                    },
                },
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Guangzhou"},
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25, "condition": "sunny"}',
            "tool_call_id": "call_1",
        },
        {
            "role": "tool",
            "content": '{"temperature": 30, "condition": "cloudy"}',
            "tool_call_id": "call_2",
        },
        {
            "role": "tool",
            "content": '{"temperature": 35, "condition": "rainy"}',
            "tool_call_id": "call_3",
        },
    ]


class LongChainTrajectory:
    """sys, user, ass(tool), tool, ass(tool:date), tool, ass(tool), tool — 3 turns"""

    TOOLS = ALL_TOOLS
    PRETOKENIZE_POSITIONS = [3, 5, 7]
    APPEND_ROLES = frozenset({"tool"})
    IS_THINKING = False
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Do a multi-step task"},
        {
            "role": "assistant",
            "content": "Step 1",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25}',
            "tool_call_id": "call_1",
        },
        {
            "role": "assistant",
            "content": "Step 2",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_date",
                        "arguments": {"timezone": "UTC"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"date": "2025-03-15", "time": "12:00:00"}',
            "tool_call_id": "call_2",
        },
        {
            "role": "assistant",
            "content": "Step 3",
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Guangzhou"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 35}',
            "tool_call_id": "call_3",
        },
    ]


class RetrySystemTrajectory:
    """sys, user, ass(tool), tool, system_retry, ass(tool), tool — 2 turns with mid-conversation system message.

    Simulates an agent that injects a system-level retry prompt when the model
    fails to produce a useful tool call on the first attempt.
    """

    TOOLS = WEATHER_TOOLS
    PRETOKENIZE_POSITIONS = [3, 6]
    APPEND_ROLES = frozenset({"tool", "system"})
    IS_THINKING = False
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Beijing and Shanghai?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25, "condition": "sunny"}',
            "tool_call_id": "call_1",
        },
        {"role": "system", "content": "You still need to check Shanghai. Please call get_weather for Shanghai."},
        {
            "role": "assistant",
            "content": "Let me check Shanghai.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Shanghai"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 30, "condition": "cloudy"}',
            "tool_call_id": "call_2",
        },
    ]


class MultiUserToolChainTrajectory:
    """sys, user1, ass(tool), tool, ass, user2, ass(tool), tool, ass(tool:date), tool

    NOTE: LinearTrajectory can carry multiple user messages when
    ``allowed_append_roles`` includes ``"user"``; this trajectory exercises
    that path.  The distribution may still deviate from the chat template
    behavior, causing high tito_session_mismatch_rate.
    """

    TOOLS = ALL_TOOLS
    PRETOKENIZE_POSITIONS = [3, 5, 7, 9]
    APPEND_ROLES = frozenset({"tool", "user"})
    IS_THINKING = False
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Beijing?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25, "condition": "sunny"}',
            "tool_call_id": "call_1",
        },
        {
            "role": "assistant",
            "content": "Beijing is 25C and sunny.",
        },
        {"role": "user", "content": "Now check Shanghai and what date is it?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Shanghai"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 30, "condition": "cloudy"}',
            "tool_call_id": "call_2",
        },
        {
            "role": "assistant",
            "content": "Shanghai is 30C. Let me check the date.",
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "get_date",
                        "arguments": {"timezone": "Asia/Shanghai"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"date": "2025-03-15", "time": "22:30:00"}',
            "tool_call_id": "call_3",
        },
    ]


class SimpleNoToolTrajectory:
    """sys, user, asst, system_reminder, user2, asst2 — no tools, with system/user append boundaries.

    Codifies the synthetic 'single_system' case the old CI used to manually
    append: at N=3 prefix ends at the first asst, append starts with the
    system_reminder, exercising the system-append boundary on a no-tool model.
    """

    TOOLS = None
    PRETOKENIZE_POSITIONS = [3, 4, 5, 6]
    APPEND_ROLES = frozenset({"user", "system"})
    IS_THINKING = False
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help?"},
        {"role": "system", "content": "Please answer in one short sentence."},
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": "Four."},
    ]


class MultiTurnNoToolTrajectory:
    """sys, user, assistant, user (no tools) — multi-turn plain conversation"""

    TOOLS = None
    PRETOKENIZE_POSITIONS = [3, 5]
    APPEND_ROLES = frozenset({"user"})
    IS_THINKING = False
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And what about Germany?"},
    ]


class MultiTurnNoToolThinkingTrajectory:
    """sys, user, assistant(reasoning_content), user (no tools) — multi-turn with thinking"""

    TOOLS = None
    PRETOKENIZE_POSITIONS = [3, 5]
    APPEND_ROLES = frozenset({"user"})
    IS_THINKING = True
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {
            "role": "assistant",
            "reasoning_content": "The user is asking about geography. The capital of France is Paris.",
            "content": "The capital of France is Paris.",
        },
        {"role": "user", "content": "And what about Germany?"},
    ]


# ---------------------------------------------------------------------------
# Thinking variants
# ---------------------------------------------------------------------------


class SingleToolThinkingTrajectory:
    """sys, user, ass(think+tool), tool, user2, ass(think+tool), tool, user3, ass — multi-role alternating with thinking.

    Codifies the synthetic 'alternating_user_tool' case the old CI used to
    manually append: prefix cuts at N=4/N=7 exercise the user-after-tool
    boundary on a thinking model.
    """

    TOOLS = WEATHER_TOOLS
    PRETOKENIZE_POSITIONS = [3, 4, 5, 6, 7, 8]
    APPEND_ROLES = frozenset({"tool", "user"})
    IS_THINKING = True
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Beijing?"},
        {
            "role": "assistant",
            "reasoning_content": "The user wants to know the weather in Beijing. I should call the get_weather function.",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing", "unit": "celsius"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25, "condition": "sunny"}',
            "tool_call_id": "call_1",
        },
        {"role": "user", "content": "Now check Shanghai too."},
        {
            "role": "assistant",
            "reasoning_content": "Now the user wants Shanghai's weather. Calling get_weather again.",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Shanghai", "unit": "celsius"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 30, "condition": "cloudy"}',
            "tool_call_id": "call_2",
        },
        {"role": "user", "content": "And tell me the date as well."},
        {
            "role": "assistant",
            "reasoning_content": "The user is asking for the date. I'll answer based on what I know.",
            "content": "Beijing is 25°C and sunny; Shanghai is 30°C and cloudy. I don't have access to the current date.",
        },
    ]


class MultiTurnThinkingTrajectory:
    """sys, user, ass(thinking+tool), tool, ass(thinking+tool), tool — 2 turns"""

    TOOLS = WEATHER_TOOLS
    PRETOKENIZE_POSITIONS = [3, 5]
    APPEND_ROLES = frozenset({"tool"})
    IS_THINKING = True
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Beijing and Shanghai?"},
        {
            "role": "assistant",
            "reasoning_content": "Let me check Beijing first.",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25, "condition": "sunny"}',
            "tool_call_id": "call_1",
        },
        {
            "role": "assistant",
            "reasoning_content": "Beijing is 25C. Now let me check Shanghai.",
            "content": "Beijing is 25C. Let me check Shanghai.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Shanghai"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 30, "condition": "cloudy"}',
            "tool_call_id": "call_2",
        },
    ]


class LongChainThinkingTrajectory:
    """sys, user, ass(thinking+tool), tool, ass(thinking+tool), tool, ass(thinking+tool), tool — 3 turns"""

    TOOLS = ALL_TOOLS
    PRETOKENIZE_POSITIONS = [3, 5, 7]
    APPEND_ROLES = frozenset({"tool"})
    IS_THINKING = True
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Do a multi-step task"},
        {
            "role": "assistant",
            "reasoning_content": "Starting step 1, checking Beijing weather.",
            "content": "Step 1",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25}',
            "tool_call_id": "call_1",
        },
        {
            "role": "assistant",
            "reasoning_content": "Got Beijing result. Now step 2, checking the date.",
            "content": "Step 2",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_date",
                        "arguments": {"timezone": "UTC"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"date": "2025-03-15", "time": "12:00:00"}',
            "tool_call_id": "call_2",
        },
        {
            "role": "assistant",
            "reasoning_content": "Got date result. Now step 3, checking Guangzhou.",
            "content": "Step 3",
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Guangzhou"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 35}',
            "tool_call_id": "call_3",
        },
    ]


class MultiUserTurnThinkingTrajectory:
    """sys, user1, ass(thinking+tool), tool, ass(thinking), user2, ass(thinking+tool), tool

    Cross-user-turn with thinking. Tests that thinking content from user turn 1
    is not compressed/modified when rendering the full conversation including
    user turn 2.
    """

    TOOLS = WEATHER_TOOLS
    PRETOKENIZE_POSITIONS = [7]
    APPEND_ROLES = frozenset({"tool", "user"})
    IS_THINKING = True
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        # --- user turn 1 ---
        {"role": "user", "content": "What's the weather in Beijing?"},
        {
            "role": "assistant",
            "reasoning_content": "User wants Beijing weather, let me check.",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25, "condition": "sunny"}',
            "tool_call_id": "call_1",
        },
        {
            "role": "assistant",
            "reasoning_content": "Beijing is 25C and sunny. I should tell the user.",
            "content": "Beijing is 25°C and sunny!",
        },
        # --- user turn 2 ---
        {"role": "user", "content": "Now check Shanghai too."},
        {
            "role": "assistant",
            "reasoning_content": "User wants Shanghai weather now. Let me call the tool.",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Shanghai"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 30, "condition": "cloudy"}',
            "tool_call_id": "call_2",
        },
    ]


# ---------------------------------------------------------------------------
# Intermediate system message variants
# ---------------------------------------------------------------------------


class IntermediateSystemTrajectory:
    """sys, user, ass(tool), tool, system, ass(tool:date), tool, system, ass(tool), tool — 3 turns with system"""

    TOOLS = ALL_TOOLS
    PRETOKENIZE_POSITIONS = [3, 6, 9]
    APPEND_ROLES = frozenset({"tool", "system"})
    IS_THINKING = False
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Do a multi-step task"},
        {
            "role": "assistant",
            "content": "Step 1",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25}',
            "tool_call_id": "call_1",
        },
        {"role": "system", "content": "Step 1 complete. Proceed to step 2."},
        {
            "role": "assistant",
            "content": "Step 2",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_date",
                        "arguments": {"timezone": "UTC"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"date": "2025-03-15", "time": "12:00:00"}',
            "tool_call_id": "call_2",
        },
        {"role": "system", "content": "Step 2 complete. Proceed to step 3."},
        {
            "role": "assistant",
            "content": "Step 3",
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Guangzhou"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 35}',
            "tool_call_id": "call_3",
        },
    ]


class IntermediateSystemThinkingTrajectory:
    """sys, user, ass(t+tool), tool, system, ass(t+tool:date), tool, system, ass(t+tool), tool — 3 turns with system+thinking"""

    TOOLS = ALL_TOOLS
    PRETOKENIZE_POSITIONS = [3, 6, 9]
    APPEND_ROLES = frozenset({"tool", "system"})
    IS_THINKING = True
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Do a multi-step task"},
        {
            "role": "assistant",
            "reasoning_content": "Starting step 1, checking Beijing weather.",
            "content": "Step 1",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 25}',
            "tool_call_id": "call_1",
        },
        {"role": "system", "content": "Step 1 complete. Proceed to step 2."},
        {
            "role": "assistant",
            "reasoning_content": "Got Beijing result. Now step 2, checking the date.",
            "content": "Step 2",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_date",
                        "arguments": {"timezone": "UTC"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"date": "2025-03-15", "time": "12:00:00"}',
            "tool_call_id": "call_2",
        },
        {"role": "system", "content": "Step 2 complete. Proceed to step 3."},
        {
            "role": "assistant",
            "reasoning_content": "Got date result. Now step 3, checking Guangzhou.",
            "content": "Step 3",
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Guangzhou"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 35}',
            "tool_call_id": "call_3",
        },
    ]


class MultiRoleSequenceTrajectory:
    """sys, user, asst+tool, tool, user2, asst+tool, system_reminder, tool, asst-final.

    Fills the {thinking=False, append_roles={tool, user, system}} matrix cell
    that GLM47's tool+user+system SUPPORTED_TEMPLATES row otherwise has no
    fixture for. Cuts exercise three boundaries: tool-append (N=3), user-after-tool
    (N=4), system-after-asst (N=6). Cuts at N=5/N=7/N=8 are intentionally not
    listed — they only exercise generation-prompt-only or repeat tool-append
    which other trajectories already cover.
    """

    TOOLS = ALL_TOOLS
    PRETOKENIZE_POSITIONS = [3, 4, 6]
    APPEND_ROLES = frozenset({"tool", "user", "system"})
    IS_THINKING = False
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Beijing today?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_w1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing", "unit": "celsius"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 22, "condition": "sunny"}',
            "tool_call_id": "call_w1",
        },
        {"role": "user", "content": "Also tell me the date in Beijing."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_d1",
                    "type": "function",
                    "function": {
                        "name": "get_date",
                        "arguments": {"timezone": "Asia/Shanghai"},
                    },
                }
            ],
        },
        {"role": "system", "content": "Please answer in one short sentence."},
        {
            "role": "tool",
            "content": '{"date": "2026-04-28"}',
            "tool_call_id": "call_d1",
        },
        {
            "role": "assistant",
            "content": "Beijing is 22°C and sunny on 2026-04-28.",
        },
    ]


# ---------------------------------------------------------------------------
# Helpers: build Trajectory and process_fn from a trajectory class
# ---------------------------------------------------------------------------


def _split_turns(messages: list[dict[str, Any]]) -> list[tuple[list[dict], dict]]:
    """Split a full message sequence into (request_messages, assistant_message) pairs.

    Each assistant message marks the end of one turn.
    """
    turns: list[tuple[list[dict], dict]] = []
    accumulated: list[dict] = []

    i = 0
    # Collect non-assistant prefix (system, user)
    while i < len(messages) and messages[i]["role"] != "assistant":
        accumulated.append(messages[i])
        i += 1

    while i < len(messages):
        assert messages[i]["role"] == "assistant", f"Expected assistant at index {i}, got {messages[i]['role']}"
        assistant_msg = messages[i]
        turns.append((list(accumulated), assistant_msg))
        accumulated.append(assistant_msg)
        i += 1

        # Collect subsequent tool (or user for multi-user) messages
        while i < len(messages) and messages[i]["role"] != "assistant":
            accumulated.append(messages[i])
            i += 1

    return turns


def build_trajectory(
    tokenizer: Any,
    trajectory_cls: type,
    chat_template: str | None = None,
) -> Trajectory:
    """Build a Trajectory from a trajectory class.

    Splits messages into turns and computes response_text for each turn
    using the diff method: render(msgs + [assistant], gen_prompt=False) - render(msgs, gen_prompt=True).

    Args:
        tokenizer: HuggingFace tokenizer.
        trajectory_cls: A trajectory class with MESSAGES and TOOLS attributes.
        chat_template: Optional chat template string. If None, uses tokenizer's default.
    """
    messages = deepcopy(trajectory_cls.MESSAGES)
    tools = trajectory_cls.TOOLS

    raw_turns = _split_turns(messages)

    def _render(msgs: list[dict], add_generation_prompt: bool) -> str:
        if chat_template is not None:
            from miles.utils.chat_template_utils.template import apply_chat_template_from_str

            return apply_chat_template_from_str(
                chat_template, msgs, add_generation_prompt=add_generation_prompt, tools=tools
            )
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt, tools=tools
        )

    turns: list[Turn] = []
    for request_msgs, assistant_msg in raw_turns:
        prompt_text = _render(request_msgs, add_generation_prompt=True)
        with_assistant_text = _render(request_msgs + [assistant_msg], add_generation_prompt=False)
        assert with_assistant_text.startswith(prompt_text), (
            f"Assistant text does not extend prompt text.\n"
            f"prompt[-100:]: {prompt_text[-100:]!r}\n"
            f"with_ass[:len(prompt)+50]: {with_assistant_text[:len(prompt_text)+50]!r}"
        )
        response_text = with_assistant_text[len(prompt_text) :]

        turns.append(
            Turn(
                request_messages=request_msgs,
                assistant_message=assistant_msg,
                response_text=response_text,
            )
        )

    return Trajectory(tools=tools, turns=turns, full_messages=messages)


def build_process_fn(
    trajectory: Trajectory,
    tokenizer: Any,
    chat_template: str | None = None,
) -> ProcessFn:
    """Build a process_fn that maps rendered prompt strings to response texts.

    Pre-computes the exact prompt string for each turn and returns the
    corresponding response_text.
    """
    tools = trajectory.tools

    def _render(msgs: list[dict]) -> str:
        if chat_template is not None:
            from miles.utils.chat_template_utils.template import apply_chat_template_from_str

            return apply_chat_template_from_str(chat_template, msgs, add_generation_prompt=True, tools=tools)
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, tools=tools)

    # Build prompt → response mapping
    prompt_response_map: dict[str, str] = {}
    for turn in trajectory.turns:
        prompt_str = _render(turn.request_messages)
        prompt_response_map[prompt_str] = turn.response_text

    def process_fn(prompt: str) -> ProcessResult:
        for expected_prompt, response_text in prompt_response_map.items():
            if prompt == expected_prompt:
                return ProcessResult(text=response_text, finish_reason="stop")
        raise ValueError(
            f"Unexpected prompt (length={len(prompt)}).\n"
            f"Known prompts: {[len(p) for p in prompt_response_map]}\n"
            f"Prompt tail: {prompt[-200:]!r}"
        )

    return process_fn


class SequentialProcessFn:
    """A process_fn that returns response texts in turn order.

    Unlike build_process_fn which does exact prompt matching, this version
    simply returns the next turn's response_text on each call. Useful for
    e2e tests where the actual prompt may differ from pre-computed prompts
    (e.g., tool_call IDs and argument formats differ between trajectory
    definitions and mock server responses).

    Call reset() between test runs to restart from the first turn.
    """

    def __init__(self, trajectory: Trajectory):
        self._response_texts = [turn.response_text for turn in trajectory.turns]
        self._call_count = 0

    def reset(self):
        self._call_count = 0

    def __call__(self, prompt: str) -> ProcessResult:
        idx = self._call_count
        if idx >= len(self._response_texts):
            raise ValueError(
                f"Sequential process_fn exhausted: called {idx + 1} times "
                f"but only {len(self._response_texts)} turns defined"
            )
        self._call_count += 1
        return ProcessResult(text=self._response_texts[idx], finish_reason="stop")
