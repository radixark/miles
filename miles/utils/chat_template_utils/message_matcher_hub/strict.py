"""Strict template-relevant message equality — the default matcher."""

from __future__ import annotations

from typing import Any

from miles.utils.chat_template_utils.message_matcher_hub.utils import (
    _TEMPLATE_RELEVANT_KEYS,
    _WIRE_ONLY_TOOL_CALL_KEYS,
    _normalize_value,
)


def _normalize_tool_calls(value: Any) -> Any:
    """Project tool_calls down to template-relevant content for comparison.

    Only keys in `_WIRE_ONLY_TOOL_CALL_KEYS` are removed; all other values remain part of history matching.

    Deliberately a comparison-time projection, NOT a repair of the incoming
    message from stored state.  Matching only decides whether a replay is
    the same history: on a match the prefix tokens come from stored
    checkpoints and records keep the raw backend response (``index``
    intact), so nothing downstream reads the replayed keys.  Filling
    missing keys from stored would also presuppose the per-call
    correspondence this comparison is itself establishing, and cannot
    handle a client that replays a different ``index`` value rather than
    none.
    """
    if not isinstance(value, list):
        return value
    return [
        ({k: v for k, v in call.items() if k not in _WIRE_ONLY_TOOL_CALL_KEYS} if isinstance(call, dict) else call)
        for call in value
    ]


def strict_message_matches(stored: dict[str, Any], new: dict[str, Any]) -> bool:
    """Compare only the fields that affect chat-template tokenization.

    External client libraries (e.g. litellm) may inject extra keys like
    ``provider_specific_fields`` into messages.  These have no effect on
    the Jinja2 chat template output, so we only compare the keys that
    templates actually read: role, content, reasoning_content, tool_calls.
    Within tool_calls, wire-only keys such as `index` are ignored for the same reason.
    """
    for key in _TEMPLATE_RELEVANT_KEYS:
        stored_value = _normalize_value(stored.get(key))
        new_value = _normalize_value(new.get(key))
        if key == "tool_calls":
            stored_value = _normalize_tool_calls(stored_value)
            new_value = _normalize_tool_calls(new_value)
        if stored_value != new_value:
            return False
    return True
