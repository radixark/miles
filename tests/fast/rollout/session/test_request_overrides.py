import pytest

from miles.rollout.session.request_overrides import validate_session_request_override_values


def test_validate_session_request_override_values_accepts_supported_shapes() -> None:
    validate_session_request_override_values(
        {
            "custom_logit_processor": ["processor", None],
            "custom_params": {"backend": True},
            "ebnf": None,
            "frequency_penalty": 0.1,
            "ignore_eos": False,
            "logit_bias": {"42": -1.5},
            "max_completion_tokens": None,
            "max_tokens": 128,
            "min_p": 0.05,
            "min_tokens": 1,
            "presence_penalty": 0,
            "regex": "answer",
            "repetition_penalty": 1.1,
            "response_format": {"type": "json_object"},
            "seed": 42,
            "stop": ["done"],
            "stop_regex": "finished",
            "stop_token_ids": [1, 2],
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.9,
        }
    )


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("frequency_penalty", True, "number"),
        ("ignore_eos", 1, "boolean"),
        ("max_tokens", 1.5, "integer or null"),
        ("regex", 7, "string or null"),
        ("custom_params", [], "object or null"),
        ("logit_bias", {"42": "high"}, "object with numeric values or null"),
        ("stop", [1], "string or list of strings or null"),
        ("custom_logit_processor", [1], "string or list of nullable strings or null"),
        ("stop_token_ids", [True], "list of integers or null"),
        ("min_tokens", None, "integer"),
    ],
)
def test_validate_session_request_override_values_rejects_invalid_shapes(
    key: str,
    value: object,
    expected: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"invalid session request override '{key}': expected {expected}, got",
    ):
        validate_session_request_override_values({key: value})
