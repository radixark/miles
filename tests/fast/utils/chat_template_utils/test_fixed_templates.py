"""Unit tests for ``resolve_fixed_chat_template`` — one template per family.

Each ``TITOTokenizer`` subclass registers a single ``FIXED_TEMPLATE``
(template path + preserve-think kwargs).  Resolution is role-independent:
``allowed_append_roles`` only gates which roles the harness may append.
"""

import os

import pytest

from miles.utils.chat_template_utils import TEMPLATE_DIR, TITOTokenizerType, resolve_fixed_chat_template
from miles.utils.chat_template_utils.tito_tokenizer import FixedTemplate, Qwen3TITOTokenizer

_EXPECTED_FIXED_TEMPLATES = {
    TITOTokenizerType.QWEN3: ("qwen3_fixed.jinja", {"clear_thinking": False}),
    TITOTokenizerType.QWEN35: ("qwen3.5_fixed.jinja", {"clear_thinking": False}),
    TITOTokenizerType.QWENNEXT: ("qwen3_thinking_2507_and_next_fixed.jinja", {"clear_thinking": False}),
    TITOTokenizerType.GLM47: (None, {"clear_thinking": False}),
    TITOTokenizerType.NEMOTRON3: (None, {"truncate_history_thinking": False}),
    TITOTokenizerType.KIMI25: ("kimi_k25_fixed.jinja", {"preserve_thinking": True}),
    TITOTokenizerType.KIMI26: (None, {"preserve_thinking": True}),
    TITOTokenizerType.MINIMAX_M25: ("minimax_m25_fixed.jinja", {"clear_thinking": False}),
    TITOTokenizerType.MINIMAX_M27: ("minimax_m27_fixed.jinja", {"clear_thinking": False}),
    TITOTokenizerType.DEEPSEEKV32: (None, {"drop_thinking": False}),
    TITOTokenizerType.DEEPSEEKV4: (None, {"drop_thinking": False}),
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


def test_default_family_has_no_fixed_template():
    with pytest.raises(ValueError, match="No FIXED_TEMPLATE registered"):
        resolve_fixed_chat_template(TITOTokenizerType.DEFAULT)


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
