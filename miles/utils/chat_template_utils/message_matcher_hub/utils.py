"""Shared matcher type alias, constants, and normalization helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

SessionMessageMatcher: TypeAlias = Callable[[dict[str, Any], dict[str, Any]], bool]

_TEMPLATE_RELEVANT_KEYS = ("role", "content", "reasoning_content", "tool_calls")

# SGLang serializes `index` on non-streaming tool calls, while accumulated
# streaming messages may omit or renumber it; no chat template reads it.
_WIRE_ONLY_TOOL_CALL_KEYS = ("index",)


def _normalize_value(value: Any) -> Any:
    """Normalize falsy sentinels that produce identical Jinja2 output.

    None, "" and [] are all falsy in Jinja2 and render the same way,
    but client libraries may interchange them (e.g. content: null vs ""
    for tool-call-only responses, or tool_calls: null vs []).

    Only collapses falsy values — non-falsy content (including whitespace
    like trailing newlines) is returned as-is.  Message boundary characters
    must be preserved exactly so they tokenize identically across turns.
    """
    if value is None or value == "" or value == []:
        return None
    return value
