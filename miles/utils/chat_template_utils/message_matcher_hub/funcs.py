"""Built-in message matchers, selector resolution, and append-only validation."""

from __future__ import annotations

import functools
from collections.abc import Collection
from typing import Any

from miles.utils.chat_template_utils.message_matcher_hub.utils import (
    _INVALID_JSON_OBJECT,
    _TEMPLATE_RELEVANT_KEYS,
    _WIRE_ONLY_TOOL_CALL_KEYS,
    SessionMessageMatcher,
    _normalize_json_object,
    _normalize_tool_calls,
    _normalize_value,
    _raw_values_match,
)


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


def _arguments_match(stored: Any, replayed: Any) -> bool:
    stored_normalized = _normalize_json_object(stored)
    replayed_normalized = _normalize_json_object(replayed)
    if stored_normalized is _INVALID_JSON_OBJECT or replayed_normalized is _INVALID_JSON_OBJECT:
        return _raw_values_match(stored, replayed)
    return stored_normalized == replayed_normalized


def _functions_match(stored: Any, replayed: Any) -> bool:
    if not isinstance(stored, dict) or not isinstance(replayed, dict):
        return _raw_values_match(stored, replayed)
    if stored.keys() != replayed.keys():
        return False
    for key in stored:
        if key == "arguments":
            if not _arguments_match(stored[key], replayed[key]):
                return False
        elif stored[key] != replayed[key]:
            return False
    return True


def _tool_call_matches(stored: Any, replayed: Any) -> bool:
    if not isinstance(stored, dict) or not isinstance(replayed, dict):
        return _raw_values_match(stored, replayed)
    stored_projected = {key: value for key, value in stored.items() if key not in _WIRE_ONLY_TOOL_CALL_KEYS}
    replayed_projected = {key: value for key, value in replayed.items() if key not in _WIRE_ONLY_TOOL_CALL_KEYS}
    if stored_projected.keys() != replayed_projected.keys():
        return False
    for key in stored_projected:
        if key == "function":
            if not _functions_match(stored_projected[key], replayed_projected[key]):
                return False
        elif stored_projected[key] != replayed_projected[key]:
            return False
    return True


def loose_tool_call_message_matches(stored: dict[str, Any], replayed: dict[str, Any]) -> bool:
    """Match strict messages plus equivalent JSON-object tool arguments.

    Compatibility superset of ``strict_message_matches``: anything strict
    accepts stays accepted, and the only new equivalence is controlled
    JSON-object representation normalization of
    ``tool_calls[].function.arguments``.  Call ``id``, ``type``,
    ``function.name``, call order, unknown extension fields and
    ``reasoning_content`` are still compared.
    """
    try:
        if strict_message_matches(stored, replayed):
            return True
    except RecursionError:
        pass
    for key in ("role", "content", "reasoning_content"):
        if _normalize_value(stored.get(key)) != _normalize_value(replayed.get(key)):
            return False
    stored_calls = _normalize_value(stored.get("tool_calls"))
    replayed_calls = _normalize_value(replayed.get("tool_calls"))
    if not isinstance(stored_calls, list) or not isinstance(replayed_calls, list):
        return _raw_values_match(stored_calls, replayed_calls)
    return len(stored_calls) == len(replayed_calls) and all(
        _tool_call_matches(left, right) for left, right in zip(stored_calls, replayed_calls, strict=True)
    )


def role_content_only_message_matches(stored: dict[str, Any], replayed: dict[str, Any]) -> bool:
    """Compare only role and content using the strict matcher's empty-value rule.

    High-risk field projection: every other message field — including whole
    ``tool_calls`` — is deliberately ignored, so the stored prefix wins for
    anything a template might read from those fields.
    """
    return all(_normalize_value(stored.get(key)) == _normalize_value(replayed.get(key)) for key in ("role", "content"))


_BUILTIN_MESSAGE_MATCHERS: dict[str, SessionMessageMatcher] = {
    "strict": strict_message_matches,
    "loose_tool_call": loose_tool_call_message_matches,
    "role_content_only": role_content_only_message_matches,
}


class SessionMessageMatcherError(Exception):
    """Raised when a configured matcher throws or returns a non-bool value."""


def _validated(matcher: SessionMessageMatcher) -> SessionMessageMatcher:
    """Enforce the matcher contract at runtime: exact bool, no exceptions.

    A truthy non-bool or a raised exception is a configuration error, never a
    match or mismatch; surface it as ``SessionMessageMatcherError`` so the
    session server maps it to HTTP 500 instead of silently deciding session
    identity from a broken matcher.
    """

    @functools.wraps(matcher)
    def validated(stored: dict[str, Any], replayed: dict[str, Any]) -> bool:
        try:
            result = matcher(stored, replayed)
        except Exception as exc:
            raise SessionMessageMatcherError("session message matcher raised an exception") from exc
        if type(result) is not bool:
            raise SessionMessageMatcherError(f"session message matcher must return bool, got {type(result).__name__}")
        return result

    return validated


def resolve_session_message_matcher(selector: str) -> SessionMessageMatcher:
    """Resolve an exact built-in alias or a synchronous dotted import path.

    Every result carries the runtime contract check (``_validated``); the
    original matcher stays reachable as ``__wrapped__``.
    """
    if selector in _BUILTIN_MESSAGE_MATCHERS:
        return _validated(_BUILTIN_MESSAGE_MATCHERS[selector])
    aliases = ", ".join(_BUILTIN_MESSAGE_MATCHERS)
    if not isinstance(selector, str) or not selector or "." not in selector:
        raise ValueError(
            f"invalid --session-message-matcher {selector!r}; use one of {aliases}, "
            f"or a dotted import path such as package.module.matcher"
        )
    try:
        from miles.utils.misc import load_function

        return _validated(load_function(selector, sync_required=True))
    except Exception as exc:
        raise ValueError(
            f"failed to resolve --session-message-matcher {selector!r}; use one of {aliases}, "
            f"or a dotted import path such as package.module.matcher: {exc}"
        ) from exc


def assert_messages_append_only_with_allowed_role(
    stored_messages: list[dict[str, Any]],
    new_messages: list[dict[str, Any]],
    allowed_append_roles: Collection[str],
    *,
    message_matcher: SessionMessageMatcher | None = None,
) -> None:
    """Assert *new_messages* is an append-only extension of *stored_messages*.

    The stored prefix must match pairwise under *message_matcher* (defaults
    to the strict template-relevant comparison), and any appended messages
    must have a role in *allowed_append_roles*.
    """
    if not stored_messages:
        return

    matcher = message_matcher if message_matcher is not None else strict_message_matches

    if len(new_messages) < len(stored_messages):
        raise ValueError(
            f"new messages ({len(new_messages)}) are fewer than stored messages ({len(stored_messages)})",
            new_messages,
            stored_messages,
        )

    for i, stored_msg in enumerate(stored_messages):
        if not matcher(stored_msg, new_messages[i]):
            diffs = {
                key: {"stored": repr(stored_msg.get(key))[:200], "new": repr(new_messages[i].get(key))[:200]}
                for key in _TEMPLATE_RELEVANT_KEYS
                if stored_msg.get(key) != new_messages[i].get(key)
            }
            raise ValueError(
                f"message mismatch at index {i} "
                f"(role: stored={stored_msg.get('role')}, new={new_messages[i].get('role')}). "
                f"Diffs: {diffs}"
            )

    for j, msg in enumerate(new_messages[len(stored_messages) :]):
        if msg.get("role") not in allowed_append_roles:
            raise ValueError(
                f"appended message at index {len(stored_messages) + j} "
                f"has role={msg.get('role')!r}, allowed={allowed_append_roles}"
            )
