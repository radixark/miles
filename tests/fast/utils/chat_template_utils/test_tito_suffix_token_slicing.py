"""Regression for token-id drift in ``TITOTokenizer._tokenize_rendered_suffix``.

Suffix ids come from the full prompt's tokenization, not a standalone re-encode
of the rendered text slice. Re-encoding the slice can insert a start-of-segment
marker that is absent in context (issue #1319).
"""

from __future__ import annotations

from typing import Any

from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer


class StartMarkerTokenizer:
    """SentencePiece-like stub: the first char of any ``encode`` call gets a
    start-of-segment variant id, differing from the same char mid-sequence."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._vocab.setdefault(("^" if i == 0 else "") + ch, len(self._vocab)) for i, ch in enumerate(text)]


class _StubTITOTokenizer(TITOTokenizer):
    """Pins the rendered prefix/suffix text so the test depends only on slicing."""

    def __init__(self, prefix_text: str, suffix_text: str) -> None:
        super().__init__(StartMarkerTokenizer())
        self._prefix_text = prefix_text
        self._suffix_text = suffix_text

    def apply_chat_template(self, messages: list[dict[str, Any]], *, add_generation_prompt, tools=None, tokenize=False):
        return self._prefix_text + (self._suffix_text if len(messages) > 1 else "")


def test_suffix_uses_in_context_ids_not_standalone_reencode():
    tito = _StubTITOTokenizer(prefix_text="hi ", suffix_text="bye")

    suffix = tito._tokenize_rendered_suffix([{"role": "system"}], [{"role": "user"}])

    assert tito._encode_text("hi ") + suffix == tito._encode_text("hi bye")
    assert suffix != tito._encode_text("bye")
