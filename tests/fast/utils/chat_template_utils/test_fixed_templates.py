"""Unit tests for ``resolve_fixed_chat_template`` — one template per family.

Each ``TITOTokenizer`` subclass registers a single ``FIXED_TEMPLATE``:
template path, required kwargs, and the role surface that renderer supports.
"""

import os

import pytest

from miles.utils.chat_template_utils import TEMPLATE_DIR, TITOTokenizerType, resolve_fixed_chat_template
from miles.utils.chat_template_utils.template import apply_chat_template_from_str
from miles.utils.chat_template_utils.tito_tokenizer import (
    ALL_APPEND_ROLES,
    DeepSeekV4TITOTokenizer,
    FixedTemplate,
    MinimaxM25TITOTokenizer,
    MinimaxM27TITOTokenizer,
    Qwen3TITOTokenizer,
    Qwen35TITOTokenizer,
    TITOTokenizer,
)

_EXPECTED_FIXED_TEMPLATES = {
    TITOTokenizerType.QWEN3: ("qwen3_fixed.jinja", {"clear_thinking": False}),
    TITOTokenizerType.QWEN35: ("qwen3.5_and_3.6_fixed.jinja", {"preserve_thinking": True}),
    TITOTokenizerType.QWEN36: ("qwen3.5_and_3.6_fixed.jinja", {"preserve_thinking": True}),
    TITOTokenizerType.QWENNEXT: ("qwen3_thinking_2507_and_next_fixed.jinja", {"clear_thinking": False}),
    TITOTokenizerType.GLM47: (None, {"clear_thinking": False}),
    TITOTokenizerType.NEMOTRON3: (None, {"truncate_history_thinking": False}),
    TITOTokenizerType.KIMI25: ("kimi_k25_fixed.jinja", {"preserve_thinking": True}),
    TITOTokenizerType.KIMI26: (None, {"preserve_thinking": True}),
    TITOTokenizerType.MINIMAX_M25: ("minimax_m25_fixed.jinja", {"clear_thinking": False}),
    TITOTokenizerType.MINIMAX_M27: ("minimax_m27_fixed.jinja", {"clear_thinking": False}),
    TITOTokenizerType.DEEPSEEKV32: (None, {"drop_thinking": False}),
    TITOTokenizerType.DEEPSEEKV4: (None, {"drop_thinking": False}),
    TITOTokenizerType.INKLING: ("inkling_fixed.jinja", {}),
}


def test_every_non_default_family_is_covered():
    # New families must register a FIXED_TEMPLATE and take a row here.
    assert set(_EXPECTED_FIXED_TEMPLATES) == set(TITOTokenizerType) - {TITOTokenizerType.DEFAULT}


@pytest.mark.parametrize(
    "tito_model", list(_EXPECTED_FIXED_TEMPLATES), ids=[t.value for t in _EXPECTED_FIXED_TEMPLATES]
)
def test_family_resolves_template_and_preserve_think_kwargs(tito_model):
    # Every family pins its preserve-think kwargs unconditionally, so renders
    # stay append-only regardless of which roles the harness appends.
    expected_template, expected_kwargs = _EXPECTED_FIXED_TEMPLATES[tito_model]
    path, kwargs = resolve_fixed_chat_template(tito_model)
    if expected_template is None:
        assert path is None
    else:
        assert path == str(TEMPLATE_DIR / expected_template)
        assert os.path.isfile(path)
    assert kwargs == expected_kwargs


def test_default_family_uses_native_template():
    assert resolve_fixed_chat_template(TITOTokenizerType.DEFAULT) == (None, {})


def test_fixed_template_defaults_to_all_roles():
    fixed = FixedTemplate()
    assert fixed.allowed_append_roles == ALL_APPEND_ROLES
    assert isinstance(fixed.allowed_append_roles, frozenset)
    assert TITOTokenizer.FIXED_TEMPLATE == fixed


def test_fixed_template_rejects_unknown_role():
    with pytest.raises(ValueError, match="Unknown FixedTemplate allowed_append_roles"):
        FixedTemplate(allowed_append_roles=frozenset({"developer"}))


@pytest.mark.parametrize(
    "tokenizer_cls",
    [DeepSeekV4TITOTokenizer, Qwen35TITOTokenizer, MinimaxM25TITOTokenizer, MinimaxM27TITOTokenizer],
)
def test_restricted_fixed_template_excludes_mid_session_system(tokenizer_cls):
    assert tokenizer_cls.FIXED_TEMPLATE.allowed_append_roles == frozenset({"tool", "user", "assistant"})


def test_string_tito_model_accepted():
    assert resolve_fixed_chat_template("qwen3") == resolve_fixed_chat_template(TITOTokenizerType.QWEN3)


def test_kwargs_are_copied_not_shared(monkeypatch):
    # Mutating the returned kwargs must not leak into the registration.
    monkeypatch.setattr(
        Qwen3TITOTokenizer,
        "FIXED_TEMPLATE",
        FixedTemplate(template=None, extra_kwargs={"clear_thinking": False}),
    )
    _path, kwargs = resolve_fixed_chat_template(TITOTokenizerType.QWEN3)
    kwargs["clear_thinking"] = True
    assert Qwen3TITOTokenizer.FIXED_TEMPLATE.extra_kwargs == {"clear_thinking": False}


def test_registered_kwargs_cannot_be_overridden():
    with pytest.raises(ValueError, match="conflicts with the value registered"):
        Qwen3TITOTokenizer(object(), chat_template_kwargs={"clear_thinking": True})


def test_qwen35_and_qwen36_share_qwen36_tool_argument_serialization():
    template_path, kwargs = resolve_fixed_chat_template(TITOTokenizerType.QWEN35)
    assert template_path is not None
    with open(template_path, encoding="utf-8") as template_file:
        chat_template = template_file.read()
    rendered = apply_chat_template_from_str(
        chat_template,
        [
            {"role": "user", "content": "call"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "arguments": {
                                "string": "value",
                                "boolean": False,
                                "number": 3,
                                "nothing": None,
                                "array": [1, 2],
                                "object": {"a": 1},
                            },
                        },
                    }
                ],
            },
        ],
        add_generation_prompt=False,
        **kwargs,
    )
    assert "<parameter=string>\nvalue\n</parameter>" in rendered
    assert "<parameter=boolean>\nfalse\n</parameter>" in rendered
    assert "<parameter=number>\n3\n</parameter>" in rendered
    assert "<parameter=nothing>\nnull\n</parameter>" in rendered
    assert "<parameter=array>\n[1, 2]\n</parameter>" in rendered
    assert '<parameter=object>\n{"a": 1}\n</parameter>' in rendered
