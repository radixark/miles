"""Sampling fields that Miles may pin for every request in a session.

Session overrides are authoritative: Miles applies them after translating each
client request, so a pinned key intentionally replaces the per-turn value. This
includes Anthropic's required ``max_tokens`` field; its client value is validated
by the Anthropic adapter but ignored when the session pins ``max_tokens``. Omit a
key from ``request_overrides`` when the client should control it per turn.
"""

from collections.abc import Callable, Mapping


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_boolean(value: object) -> bool:
    return isinstance(value, bool)


def _is_string(value: object) -> bool:
    return isinstance(value, str)


def _is_mapping(value: object) -> bool:
    return isinstance(value, Mapping)


def _is_logit_bias(value: object) -> bool:
    return isinstance(value, Mapping) and all(isinstance(key, str) and _is_number(bias) for key, bias in value.items())


def _is_string_or_string_list(value: object) -> bool:
    return isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value))


def _is_string_or_nullable_string_list(value: object) -> bool:
    return isinstance(value, str) or (
        isinstance(value, list) and all(item is None or isinstance(item, str) for item in value)
    )


def _is_integer_list(value: object) -> bool:
    return isinstance(value, list) and all(_is_integer(item) for item in value)


_OverrideValidator = tuple[str, Callable[[object], bool], bool]
_SESSION_REQUEST_OVERRIDE_VALIDATORS: dict[str, _OverrideValidator] = {
    "custom_logit_processor": ("string or list of nullable strings", _is_string_or_nullable_string_list, True),
    "custom_params": ("object", _is_mapping, True),
    "ebnf": ("string", _is_string, True),
    "frequency_penalty": ("number", _is_number, False),
    "ignore_eos": ("boolean", _is_boolean, False),
    "logit_bias": ("object with numeric values", _is_logit_bias, True),
    "max_completion_tokens": ("integer", _is_integer, True),
    "max_tokens": ("integer", _is_integer, True),
    "min_p": ("number", _is_number, True),
    "min_tokens": ("integer", _is_integer, False),
    "presence_penalty": ("number", _is_number, False),
    "regex": ("string", _is_string, True),
    "repetition_penalty": ("number", _is_number, True),
    "response_format": ("object", _is_mapping, True),
    "seed": ("integer", _is_integer, True),
    "stop": ("string or list of strings", _is_string_or_string_list, True),
    "stop_regex": ("string or list of strings", _is_string_or_string_list, True),
    "stop_token_ids": ("list of integers", _is_integer_list, True),
    "temperature": ("number", _is_number, True),
    "top_k": ("integer", _is_integer, True),
    "top_p": ("number", _is_number, True),
}

SESSION_REQUEST_OVERRIDE_KEYS = frozenset(_SESSION_REQUEST_OVERRIDE_VALIDATORS)


def validate_session_request_override_values(request_overrides: Mapping[str, object]) -> None:
    """Reject override values whose JSON shape cannot satisfy SGLang's request model."""
    for key, value in request_overrides.items():
        expected, validator, nullable = _SESSION_REQUEST_OVERRIDE_VALIDATORS[key]
        if (value is None and nullable) or validator(value):
            continue
        if nullable:
            expected = f"{expected} or null"
        raise ValueError(f"invalid session request override {key!r}: expected {expected}, got {type(value).__name__}")
