"""Regression for token-id drift in ``TITOTokenizer._tokenize_rendered_suffix``.

Suffix ids come from the full prompt's tokenization, not a standalone re-encode
of the rendered text slice. Re-encoding the slice can insert a start-of-segment
marker that is absent in context (issue #1319).
"""

from __future__ import annotations

from typing import Any

import pytest

from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer


class StartMarkerTokenizer:
    """SentencePiece-like stub: the first char of any ``encode`` call gets a
    start-of-segment variant id, differing from the same char mid-sequence."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._vocab.setdefault(("^" if i == 0 else "") + ch, len(self._vocab)) for i, ch in enumerate(text)]


class CharTokenizer:
    """Context-free stub: each character is its own token."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]


class JunctionMergeTokenizer:
    """Merges a trailing space with the next character, so encode(prefix) is
    not a token-prefix of encode(prefix + suffix)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text[i] == " " and i + 1 < len(text):
                ids.append(1000 + ord(text[i + 1]))
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids


class _StubTITOTokenizer(TITOTokenizer):
    """Pins the rendered prefix/suffix text so the test depends only on slicing."""

    def __init__(self, tokenizer: Any, prefix_text: str, suffix_text: str) -> None:
        super().__init__(tokenizer)
        self._prefix_text = prefix_text
        self._suffix_text = suffix_text

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt,
        tools=None,
        tokenize=False,
    ):
        return self._prefix_text + (self._suffix_text if len(messages) > 1 else "")


def test_suffix_uses_in_context_ids_not_standalone_reencode():
    tito = _StubTITOTokenizer(StartMarkerTokenizer(), prefix_text="hi ", suffix_text="bye")

    suffix = tito._tokenize_rendered_suffix([{"role": "system"}], [{"role": "user"}])

    assert tito._encode_text("hi ") + suffix == tito._encode_text("hi bye")
    assert suffix != tito._encode_text("bye")


def test_suffix_matches_standalone_encode_when_no_boundary_span():
    tito = _StubTITOTokenizer(CharTokenizer(), prefix_text="hi ", suffix_text="bye")

    suffix = tito._tokenize_rendered_suffix([{"role": "system"}], [{"role": "user"}])

    assert suffix == tito._encode_text("bye")
    assert tito._encode_text("hi ") + suffix == tito._encode_text("hi bye")


def test_raises_when_prefix_ids_are_not_a_prefix_of_full_ids():
    tito = _StubTITOTokenizer(JunctionMergeTokenizer(), prefix_text="hi ", suffix_text="bye")

    with pytest.raises(ValueError, match="token-id suffix diff failed"):
        tito._tokenize_rendered_suffix([{"role": "system"}], [{"role": "user"}])
