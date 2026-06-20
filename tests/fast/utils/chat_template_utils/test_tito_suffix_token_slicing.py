"""Regression test for token drift in ``TITOTokenizer._tokenize_rendered_suffix`` (issue #1319).

The suffix ids for an appended segment are taken from the full prompt's own
tokenization (``full_ids[len(prefix_ids):]``), not by re-encoding the rendered
text slice on its own. A standalone re-encode can add a start-of-segment marker
(e.g. SentencePiece's leading ``▁``) that is absent in context, so ``prefix +
suffix`` would no longer reproduce the full prompt.

``sglang`` is stubbed before importing the target module, mirroring the repo's
fast-test pattern for avoiding heavy runtime deps not needed at unit level.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pydantic


def _install_sglang_stub() -> None:
    def _reg(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.entrypoints",
        "sglang.srt.entrypoints.openai",
        "sglang.srt.entrypoints.openai.protocol",
    ):
        if name not in sys.modules:
            _reg(name)

    class Tool(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow")
        type: str = "function"
        function: dict = {}

    sys.modules["sglang.srt.entrypoints.openai.protocol"].Tool = Tool
    enc4 = _reg("sglang.srt.entrypoints.openai.encoding_dsv4")
    enc4.encode_messages = lambda *a, **k: []
    enc32 = _reg("sglang.srt.entrypoints.openai.encoding_dsv32")
    enc32.encode_messages = lambda *a, **k: []
    sys.modules["sglang.srt.entrypoints.openai"].encoding_dsv4 = enc4
    sys.modules["sglang.srt.entrypoints.openai"].encoding_dsv32 = enc32


_install_sglang_stub()

from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer  # noqa: E402


class StartMarkerTokenizer:
    """SentencePiece-like stub: the first char of any ``encode`` call gets a
    start-of-segment variant id, differing from the same char mid-sequence. So
    re-encoding a text slice on its own drifts from the in-context ids while the
    token counts stay aligned -- the shape of issue #1319."""

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

    def render_messages(self, messages: list[dict[str, Any]], *, add_generation_prompt, tools=None, tokenize=False):
        return self._prefix_text + (self._suffix_text if len(messages) > 1 else "")


def test_suffix_uses_in_context_ids_not_standalone_reencode():
    tito = _StubTITOTokenizer(prefix_text="hi ", suffix_text="bye")

    suffix = tito._tokenize_rendered_suffix([{"role": "system"}], [{"role": "user"}])

    # The suffix is the full prompt's own tail, so prefix + suffix == full prompt.
    assert tito._encode_text("hi ") + suffix == tito._encode_text("hi bye")
    # ...and it differs from re-encoding the text slice, which drifts (start marker).
    assert suffix != tito._encode_text("bye")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
