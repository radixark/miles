"""Tests for TITOTokenizer auto-detection, create_comparator(), and subclass configuration."""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from miles.utils.chat_template_utils.tito_tokenizer import (
    GLM47TITOTokenizer,
    Qwen3TITOTokenizer,
    TITOTokenizer,
    TITOTokenizerType,
    get_tito_tokenizer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TOK_CACHE: dict[str, AutoTokenizer] = {}


def _get_tokenizer(model_id: str, trust_remote_code: bool = True) -> AutoTokenizer:
    if model_id not in _TOK_CACHE:
        _TOK_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return _TOK_CACHE[model_id]


@pytest.fixture
def qwen3_tito() -> Qwen3TITOTokenizer:
    tok = _get_tokenizer("Qwen/Qwen3-4B")
    return Qwen3TITOTokenizer(tok)


@pytest.fixture
def glm47_tito() -> GLM47TITOTokenizer:
    tok = _get_tokenizer("zai-org/GLM-4.7-Flash")
    return GLM47TITOTokenizer(tok)


@pytest.fixture
def default_tito() -> TITOTokenizer:
    tok = _get_tokenizer("Qwen/Qwen3-4B")
    return TITOTokenizer(tok)


# ---------------------------------------------------------------------------
# Auto-detected assistant_start_str
# ---------------------------------------------------------------------------


class TestAssistantStartStr:
    def test_qwen3_hardcoded(self, qwen3_tito: Qwen3TITOTokenizer):
        assert qwen3_tito._assistant_start_str == "<|im_start|>assistant"

    def test_glm47_hardcoded(self, glm47_tito: GLM47TITOTokenizer):
        assert glm47_tito._assistant_start_str == "<|assistant|>"

    def test_base_class_default_is_none(self, default_tito: TITOTokenizer):
        """Base class has no hardcoded default; pass assistant_start_str explicitly if needed."""
        assert default_tito._assistant_start_str is None

    def test_explicit_override(self):
        tok = _get_tokenizer("Qwen/Qwen3-4B")
        tito = TITOTokenizer(tok, assistant_start_str="custom")
        assert tito._assistant_start_str == "custom"

    def test_propagated_to_comparator(self, qwen3_tito: Qwen3TITOTokenizer):
        comp = qwen3_tito.create_comparator()
        assert comp._assistant_start_str == qwen3_tito._assistant_start_str


# ---------------------------------------------------------------------------
# create_comparator
# ---------------------------------------------------------------------------


class TestCreateComparator:
    def test_returns_new_instance(self, qwen3_tito: Qwen3TITOTokenizer):
        """Each call creates a fresh comparator."""
        comp1 = qwen3_tito.create_comparator()
        comp2 = qwen3_tito.create_comparator()
        assert comp1 is not comp2


# ---------------------------------------------------------------------------
# Subclass init — boundary token IDs
# ---------------------------------------------------------------------------


class TestSubclassInit:
    def test_qwen3_boundary_tokens(self, qwen3_tito: Qwen3TITOTokenizer):
        tok = qwen3_tito.tokenizer
        assert qwen3_tito._im_end_id == tok.convert_tokens_to_ids("<|im_end|>")
        assert qwen3_tito._newline_id == tok.encode("\n", add_special_tokens=False)[0]

    def test_glm47_boundary_tokens(self, glm47_tito: GLM47TITOTokenizer):
        tok = glm47_tito.tokenizer
        assert glm47_tito._user_id == tok.convert_tokens_to_ids("<|user|>")
        assert glm47_tito._observation_id == tok.convert_tokens_to_ids("<|observation|>")
        assert glm47_tito._ambiguous_boundary_ids == {glm47_tito._user_id, glm47_tito._observation_id}


# ---------------------------------------------------------------------------
# trailing_token_ids / max_trim_tokens
# ---------------------------------------------------------------------------


class TestTrailingConfig:
    def test_default_no_trailing(self, default_tito: TITOTokenizer):
        assert default_tito.trailing_token_ids == frozenset()
        assert default_tito.max_trim_tokens == 0

    def test_qwen3_trailing_newline(self, qwen3_tito: Qwen3TITOTokenizer):
        assert qwen3_tito.trailing_token_ids == frozenset({qwen3_tito._newline_id})

    def test_glm47_trailing_boundary(self, glm47_tito: GLM47TITOTokenizer):
        assert glm47_tito.trailing_token_ids == frozenset({glm47_tito._user_id, glm47_tito._observation_id})
        assert glm47_tito.max_trim_tokens == 1


# ---------------------------------------------------------------------------
# get_tito_tokenizer factory — assistant config propagated
# ---------------------------------------------------------------------------


class TestFactory:
    def test_factory_qwen3(self):
        tok = _get_tokenizer("Qwen/Qwen3-4B")
        tito = get_tito_tokenizer(tok, tokenizer_type="qwen3")
        assert isinstance(tito, Qwen3TITOTokenizer)
        assert tito._assistant_start_str == "<|im_start|>assistant"

    def test_factory_glm47(self):
        tok = _get_tokenizer("zai-org/GLM-4.7-Flash")
        tito = get_tito_tokenizer(tok, tokenizer_type="glm47")
        assert isinstance(tito, GLM47TITOTokenizer)
        assert tito._assistant_start_str == "<|assistant|>"

    def test_factory_default(self):
        tok = _get_tokenizer("Qwen/Qwen3-4B")
        tito = get_tito_tokenizer(tok, tokenizer_type="default")
        assert isinstance(tito, TITOTokenizer)
        assert tito._assistant_start_str is None

    def test_factory_explicit_override(self):
        tok = _get_tokenizer("Qwen/Qwen3-4B")
        tito = get_tito_tokenizer(tok, tokenizer_type="qwen3", assistant_start_str="custom")
        assert tito._assistant_start_str == "custom"

    def test_factory_enum_input(self):
        tok = _get_tokenizer("Qwen/Qwen3-4B")
        tito = get_tito_tokenizer(tok, tokenizer_type=TITOTokenizerType.QWEN3)
        assert isinstance(tito, Qwen3TITOTokenizer)

    def test_factory_invalid_type(self):
        tok = _get_tokenizer("Qwen/Qwen3-4B")
        with pytest.raises(ValueError):
            get_tito_tokenizer(tok, tokenizer_type="nonexistent")
