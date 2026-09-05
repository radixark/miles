from copy import deepcopy

import pytest

from miles.utils.arg_resolution import ArgResolutionError
from miles.utils.chat_template_utils.reasoning import QWEN38_REASONING, ReasoningTemplateConfig


def test_profile_requires_an_active_default_and_freezes_allowed_efforts():
    efforts = {"low", "xhigh"}
    profile = ReasoningTemplateConfig("enable_thinking", "xhigh", efforts)
    efforts.remove("xhigh")
    assert profile.active_efforts == frozenset({"low", "xhigh"})
    with pytest.raises(ValueError, match="default_effort"):
        ReasoningTemplateConfig("enable_thinking", "none", efforts)


@pytest.mark.parametrize(
    ("launch", "request_body", "expected"),
    [
        ({}, {}, {"reasoning_effort": "xhigh"}),
        ({"reasoning_effort": "low"}, {}, {"reasoning_effort": "low"}),
        ({"reasoning_effort": None}, {}, {"reasoning_effort": "xhigh"}),
        ({"reasoning_effort": "none"}, {}, {"reasoning_effort": "xhigh", "enable_thinking": False}),
        (
            {"reasoning_effort": "none", "enable_thinking": True},
            {},
            {"reasoning_effort": "xhigh", "enable_thinking": True},
        ),
        (
            {"reasoning_effort": "medium"},
            {"reasoning_effort": "none"},
            {"reasoning_effort": "medium", "enable_thinking": False},
        ),
        (
            {"enable_thinking": False},
            {"reasoning_effort": "low"},
            {"reasoning_effort": "low", "enable_thinking": True},
        ),
        ({}, {"reasoning": {"enabled": True}}, {"reasoning_effort": "xhigh", "enable_thinking": True}),
        (
            {},
            {"reasoning_effort": "none", "reasoning": {"enabled": True}},
            {"reasoning_effort": "xhigh", "enable_thinking": False},
        ),
        (
            {},
            {"reasoning_effort": "low", "reasoning": {"effort": "none", "reasoning_effort": "medium"}},
            {"reasoning_effort": "low", "enable_thinking": False},
        ),
        (
            {},
            {"reasoning": {"effort": "none", "reasoning_effort": "low"}},
            {"reasoning_effort": "xhigh", "enable_thinking": False},
        ),
        (
            {},
            {"reasoning_effort": "none", "reasoning": {"effort": "low"}},
            {"reasoning_effort": "low", "enable_thinking": True},
        ),
        (
            {},
            {"reasoning_effort": "low", "reasoning": {"effort": None, "reasoning_effort": "medium"}},
            {"reasoning_effort": "medium", "enable_thinking": True},
        ),
        (
            {},
            {"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "none"}},
            {"reasoning_effort": "low", "enable_thinking": False},
        ),
        (
            {},
            {"reasoning_effort": "none", "chat_template_kwargs": {"reasoning_effort": "medium"}},
            {"reasoning_effort": "medium", "enable_thinking": True},
        ),
        (
            {},
            {"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": None}},
            {"reasoning_effort": "low", "enable_thinking": True},
        ),
        ({}, {"reasoning": {"enabled": False}}, {"reasoning_effort": "xhigh"}),
        ({}, {"reasoning": {"enabled": False, "enable": True}}, {"reasoning_effort": "xhigh"}),
        ({}, {"reasoning": {"enabled": None, "enable": True}}, {"reasoning_effort": "xhigh", "enable_thinking": True}),
        ({}, {"reasoning": {"effort": None, "reasoning_effort": None}}, {"reasoning_effort": "xhigh"}),
    ],
)
def test_qwen_reasoning_authority(launch, request_body, expected):
    original = deepcopy((launch, request_body))
    assert QWEN38_REASONING.resolve(launch, request_body) == expected
    assert (launch, request_body) == original


@pytest.mark.parametrize("alias", ["enabled", "enable"])
@pytest.mark.parametrize(
    ("raw", "enabled"),
    [
        (" YES ", True),
        ("on", True),
        ("1", True),
        ("True", True),
        ("y", True),
        (1, True),
        ("false", False),
        ("0", False),
        ("anything", False),
        (False, False),
        (0, False),
        (None, False),
    ],
)
def test_object_enabled_normalization(alias, raw, enabled):
    expected = {"reasoning_effort": "xhigh"}
    if enabled:
        expected["enable_thinking"] = True
    assert QWEN38_REASONING.resolve({}, {"reasoning": {alias: raw}}) == expected


@pytest.mark.parametrize("raw", [None, False, True, "true", "false", "yes", 0, 1])
def test_direct_toggles_remain_raw_at_launch_and_nested_precedence(raw):
    launch = QWEN38_REASONING.resolve({"enable_thinking": raw}, {})
    nested = QWEN38_REASONING.resolve(
        {"enable_thinking": False},
        {"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": raw}},
    )
    assert launch["enable_thinking"] == raw
    assert type(launch["enable_thinking"]) is type(raw)
    assert nested["enable_thinking"] == raw
    assert type(nested["enable_thinking"]) is type(raw)
    assert nested["reasoning_effort"] == "low"


@pytest.mark.parametrize("invalid", ["high", "minimal", "max", "LOW", " low ", "", 1, 1.0, True, [], {}])
@pytest.mark.parametrize("alias", ["launch", "top", "object_effort", "object_alias", "nested"])
def test_every_raw_effort_alias_is_validated_even_when_it_loses(alias, invalid):
    launch = {"reasoning_effort": "low"}
    request = {
        "reasoning_effort": "medium",
        "reasoning": {"effort": "none", "reasoning_effort": "low"},
        "chat_template_kwargs": {"reasoning_effort": "xhigh", "enable_thinking": False},
    }
    if alias == "launch":
        launch["reasoning_effort"] = invalid
    elif alias == "top":
        request["reasoning_effort"] = invalid
    elif alias == "object_effort":
        request["reasoning"]["effort"] = invalid
    elif alias == "object_alias":
        request["reasoning"]["reasoning_effort"] = invalid
    else:
        request["chat_template_kwargs"]["reasoning_effort"] = invalid
    with pytest.raises(ArgResolutionError, match="Invalid reasoning effort"):
        QWEN38_REASONING.resolve(launch, request)


def test_unrelated_template_kwargs_are_preserved_without_mutation():
    launch = {"preserve_thinking": True, "add_vision_id": False, "custom": {"items": [1]}}
    request = {"chat_template_kwargs": {"add_vision_id": True, "custom": {"items": [2]}}}
    assert QWEN38_REASONING.resolve(launch, request) == {
        "preserve_thinking": True,
        "add_vision_id": True,
        "custom": {"items": [2]},
        "reasoning_effort": "xhigh",
    }
    assert launch["custom"] == {"items": [1]}
    assert request["chat_template_kwargs"]["custom"] == {"items": [2]}
