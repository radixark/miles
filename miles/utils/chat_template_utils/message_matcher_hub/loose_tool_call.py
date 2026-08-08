"""Loose matcher: strict equality plus JSON-object tool-argument equivalence."""

from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation
from typing import Any

from miles.utils.chat_template_utils.message_matcher_hub.strict import strict_message_matches
from miles.utils.chat_template_utils.message_matcher_hub.utils import _WIRE_ONLY_TOOL_CALL_KEYS, _normalize_value

_INVALID_JSON_OBJECT = object()


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
    """Convert a parsed JSON value into a type-tagged, key-sorted form.

    Tagging keeps JSON types apart under Python equality (True vs 1,
    1 vs 1.0 as Decimal-exact numbers, "1" vs 1), sorts object keys, and
    preserves array order — exactly the representation equivalence
    ``loose_tool_call`` promises and nothing more.
    """
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
    """Normalize an ``arguments`` value to a comparable JSON-object form.

    None, "" and a valid empty-object spelling all map to the empty object.
    Strings must parse as a JSON object without duplicate keys, NaN or
    Infinity; dicts must be JSON-compatible.  Everything else returns
    ``_INVALID_JSON_OBJECT`` so the caller falls back to raw comparison.
    """
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
    """Type-sensitive structural equality for non-normalizable values."""
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
