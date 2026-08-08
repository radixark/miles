"""Shared matcher building blocks: type alias, constants, and normalization.

Everything here is representation-level and policy-free, so custom matchers
loaded via ``--session-message-matcher`` can reuse it alongside the built-ins.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any, TypeAlias

SessionMessageMatcher: TypeAlias = Callable[[dict[str, Any], dict[str, Any]], bool]

_TEMPLATE_RELEVANT_KEYS = ("role", "content", "reasoning_content", "tool_calls")

# SGLang serializes `index` on non-streaming tool calls, while accumulated
# streaming messages may omit or renumber it; no chat template reads it.
_WIRE_ONLY_TOOL_CALL_KEYS = ("index",)

_INVALID_JSON_OBJECT = object()


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
