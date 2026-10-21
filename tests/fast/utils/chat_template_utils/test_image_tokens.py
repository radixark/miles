"""The per-family image-token table (``ImageTokens``): which TITO families
carry images, with which markers and placeholder, and that the family's
template and tokenizer agree with the table. The real-tokenizer cases skip
when the checkpoint is not in the local cache; ``MILES_TEST_QWEN38_TOKENIZER``
and ``MILES_TEST_GLM53_TOKENIZER`` point them at a local directory."""

from __future__ import annotations

import os

import pytest

from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.chat_template_utils.tito_tokenizer import (
    GLM_V_IMAGE_TOKENS,
    QWEN3_VL_IMAGE_TOKENS,
    ImageTokens,
    TITOTokenizer,
    TITOTokenizerType,
    image_token_table,
)

QWEN38 = os.environ.get("MILES_TEST_QWEN38_TOKENIZER", "Qwen/Qwen3.8-27B")
GLM53 = os.environ.get("MILES_TEST_GLM53_TOKENIZER", "zai-org/GLM-5.3-Flash")

VISION_FAMILIES = {
    TITOTokenizerType.QWEN35: QWEN3_VL_IMAGE_TOKENS,
    TITOTokenizerType.QWEN36: QWEN3_VL_IMAGE_TOKENS,
    TITOTokenizerType.QWEN38_SMALL: QWEN3_VL_IMAGE_TOKENS,
    TITOTokenizerType.QWEN4_EXP: QWEN3_VL_IMAGE_TOKENS,
    TITOTokenizerType.GLM53: GLM_V_IMAGE_TOKENS,
}


def _load(model: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 - no network or no cache
        pytest.skip(f"{model} tokenizer unavailable: {e}")


def _image_message(n_images: int) -> list[dict]:
    parts = [{"type": "text", "text": "look"}]
    for i in range(n_images):
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,AAA{i}"}})
    return [{"role": "user", "content": parts}]


class TestTheTable:
    def test_every_family_is_listed_and_only_vision_families_carry_images(self):
        table = image_token_table()
        assert set(table) == set(TITOTokenizerType)
        assert {t for t, tokens in table.items() if tokens is not None} == set(VISION_FAMILIES)
        for family, expected in VISION_FAMILIES.items():
            assert table[family] is expected

    def test_both_known_families_expand_one_placeholder_between_markers(self):
        assert QWEN3_VL_IMAGE_TOKENS == ImageTokens("<|vision_start|>", "<|image_pad|>", "<|vision_end|>", True)
        assert GLM_V_IMAGE_TOKENS == ImageTokens("<|begin_of_image|>", "<|image|>", "<|end_of_image|>", True)
        assert ImageTokens(None, "<img>", None).tokens == ("<img>",)

    def test_a_text_family_has_no_placeholder(self):
        class FakeTokenizer:
            unk_token_id = 0

            def convert_tokens_to_ids(self, token):
                return 0

        tito = TITOTokenizer(FakeTokenizer())
        assert tito.image_tokens is None and tito.image_token_ids() is None and tito.image_pad_id() is None

    def test_a_family_whose_tokenizer_lacks_its_tokens_fails_loudly(self):
        class TokenizerWithoutImageTokens:
            unk_token_id = 0

            def convert_tokens_to_ids(self, token):
                return 0

        tito = TITOTokenizer(TokenizerWithoutImageTokens())
        tito.image_tokens = GLM_V_IMAGE_TOKENS
        with pytest.raises(ValueError, match="<\\|begin_of_image\\|>"):
            tito.image_token_ids()


class _FamilyCase:
    """One vision family on its real tokenizer: the table's ids resolve, and
    the family's template writes exactly the table's tokens once per image,
    in order (one placeholder per image, which the engine expands)."""

    model: str
    family: str
    expected: ImageTokens

    @pytest.fixture(scope="class")
    def tito(self):
        return get_tito_tokenizer(_load(self.model), tokenizer_type=self.family)

    def test_the_tokenizer_has_every_token_of_the_table(self, tito):
        ids = tito.image_token_ids()
        assert set(ids) == {"begin", "placeholder", "end"}
        assert len(set(ids.values())) == 3
        assert tito.image_pad_id() == ids["placeholder"] == tito.tokenizer.convert_tokens_to_ids(self.expected.placeholder)
        assert tito.image_tokens is self.expected

    @pytest.mark.parametrize("n_images", [1, 2])
    def test_the_template_writes_the_table_tokens_once_per_image(self, tito, n_images):
        ids = tito.image_token_ids()
        prompt = tito.apply_chat_template(_image_message(n_images), add_generation_prompt=True, tokenize=True)
        per_image = [ids["begin"], ids["placeholder"], ids["end"]]
        image_ids = [t for t in prompt if t in set(per_image)]
        assert image_ids == per_image * n_images


class TestQwen3VL(_FamilyCase):
    model = QWEN38
    family = "qwen38small"
    expected = QWEN3_VL_IMAGE_TOKENS


class TestGLM53(_FamilyCase):
    model = GLM53
    family = "glm53"
    expected = GLM_V_IMAGE_TOKENS
