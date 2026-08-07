"""Session message matching policies and selector resolution."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection
from decimal import Decimal, InvalidOperation
from typing import Any, TypeAlias

SessionMessageMatcher: TypeAlias = Callable[[dict[str, Any], dict[str, Any]], bool]

_TEMPLATE_RELEVANT_KEYS = ("role", "content", "reasoning_content", "tool_calls")

# SGLang serializes `index` on non-streaming tool calls, while accumulated
# streaming messages may omit or renumber it; no chat template reads it.
_WIRE_ONLY_TOOL_CALL_KEYS = ("index",)

_INVALID_JSON_OBJECT = object()


def _normalize_value(value: Any) -> Any:
    """Collapse the falsy sentinels that produce identical Jinja2 output."""
    if value is None or value == "" or value == []:
        return None
    return value


def _normalize_tool_calls(value: Any) -> Any:
    """Remove wire-only fields without mutating the supplied tool calls."""
    if not isinstance(value, list):
        return value
    return [
        (
            {key: item for key, item in call.items() if key not in _WIRE_ONLY_TOOL_CALL_KEYS}
            if isinstance(call, dict)
            else call
        )
        for call in value
    ]


def message_matches(stored: dict[str, Any], new: dict[str, Any]) -> bool:
    """Compare fields that affect the current chat-template tokenization."""
    for key in _TEMPLATE_RELEVANT_KEYS:
        stored_value = _normalize_value(stored.get(key))
        new_value = _normalize_value(new.get(key))
        if key == "tool_calls":
            stored_value = _normalize_tool_calls(stored_value)
            new_value = _normalize_tool_calls(new_value)
        if stored_value != new_value:
            return False
    return True


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_json_number(raw: str) -> Decimal:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"invalid JSON number: {raw!r}") from exc
    if not value.is_finite():
        raise ValueError(f"non-finite JSON number: {raw!r}")
    return value


def _reject_json_constant(raw: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {raw!r}")


def _tag_json_value(value: Any, *, allow_decimal: bool) -> tuple[Any, ...]:
    value_type = type(value)
    if value is None:
        return ("null",)
    if value_type is bool:
        return ("boolean", value)
    if value_type is str:
        return ("string", value)
    if value_type is int:
        return ("number", Decimal(value))
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return ("number", Decimal(str(value)))
    if value_type is Decimal:
        if not allow_decimal or not value.is_finite():
            raise ValueError("value is not a JSON-compatible number")
        return ("number", value)
    if value_type is list:
        return ("array", tuple(_tag_json_value(item, allow_decimal=allow_decimal) for item in value))
    if value_type is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        return (
            "object",
            tuple(
                sorted(
                    (
                        key,
                        _tag_json_value(item, allow_decimal=allow_decimal),
                    )
                    for key, item in value.items()
                )
            ),
        )
    raise ValueError(f"value of type {value_type.__name__} is not JSON-compatible")


def _normalize_json_object(value: Any) -> tuple[Any, ...] | object:
    if value is None or value == "":
        return ("object", ())
    parsed = value
    if type(value) is str:
        try:
            parsed = json.loads(
                value,
                object_pairs_hook=_reject_duplicate_object_keys,
                parse_int=Decimal,
                parse_float=_parse_json_number,
                parse_constant=_reject_json_constant,
            )
        except (OverflowError, RecursionError, TypeError, ValueError, json.JSONDecodeError):
            return _INVALID_JSON_OBJECT
    if type(parsed) is not dict:
        return _INVALID_JSON_OBJECT
    try:
        return _tag_json_value(parsed, allow_decimal=type(value) is str)
    except (RecursionError, TypeError, ValueError):
        return _INVALID_JSON_OBJECT


def _raw_values_match(stored: Any, replayed: Any) -> bool:
    try:
        if type(stored) is not type(replayed):
            return False
        if isinstance(stored, list):
            return len(stored) == len(replayed) and all(
                _raw_values_match(left, right) for left, right in zip(stored, replayed, strict=True)
            )
        if isinstance(stored, dict):
            return stored.keys() == replayed.keys() and all(
                _raw_values_match(stored[key], replayed[key]) for key in stored
            )
        return stored == replayed
    except RecursionError:
        return False


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
    """Match strict messages plus equivalent JSON-object tool arguments."""
    try:
        if message_matches(stored, replayed):
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
    """Compare only role and content using the strict matcher's empty-value rule."""
    return all(_normalize_value(stored.get(key)) == _normalize_value(replayed.get(key)) for key in ("role", "content"))


_BUILTIN_MESSAGE_MATCHERS: dict[str, SessionMessageMatcher] = {
    "strict": message_matches,
    "loose_tool_call": loose_tool_call_message_matches,
    "role_content_only": role_content_only_message_matches,
}


def resolve_session_message_matcher(selector: str) -> SessionMessageMatcher:
    """Resolve an exact built-in alias or a synchronous dotted import path."""
    if selector in _BUILTIN_MESSAGE_MATCHERS:
        return _BUILTIN_MESSAGE_MATCHERS[selector]
    aliases = ", ".join(_BUILTIN_MESSAGE_MATCHERS)
    if not isinstance(selector, str) or not selector or "." not in selector:
        raise ValueError(
            f"invalid --session-message-matcher {selector!r}; use one of {aliases}, or a dotted import path such as package.module.matcher"
        )
    try:
        from miles.utils.misc import load_function

        return load_function(selector, sync_required=True)
    except Exception as exc:
        raise ValueError(
            f"failed to resolve --session-message-matcher {selector!r}; use one of {aliases}, or a dotted import path such as package.module.matcher: {exc}"
        ) from exc


def assert_messages_append_only_with_allowed_role(
    stored_messages: list[dict[str, Any]],
    new_messages: list[dict[str, Any]],
    allowed_append_roles: Collection[str],
) -> None:
    """Require a strict stored prefix and allowed roles in the appended suffix."""
    if not stored_messages:
        return
    if len(new_messages) < len(stored_messages):
        raise ValueError(
            f"new messages ({len(new_messages)}) are fewer than stored messages ({len(stored_messages)})",
            new_messages,
            stored_messages,
        )
    for index, stored_message in enumerate(stored_messages):
        replayed_message = new_messages[index]
        if not message_matches(stored_message, replayed_message):
            diffs = {
                key: {
                    "stored": repr(stored_message.get(key))[:200],
                    "new": repr(replayed_message.get(key))[:200],
                }
                for key in _TEMPLATE_RELEVANT_KEYS
                if stored_message.get(key) != replayed_message.get(key)
            }
            raise ValueError(
                f"message mismatch at index {index} (role: stored={stored_message.get('role')}, new={replayed_message.get('role')}). Diffs: {diffs}"
            )
    for offset, message in enumerate(new_messages[len(stored_messages) :]):
        if message.get("role") not in allowed_append_roles:
            raise ValueError(
                f"appended message at index {len(stored_messages) + offset} has role={message.get('role')!r}, allowed={allowed_append_roles}"
            )
