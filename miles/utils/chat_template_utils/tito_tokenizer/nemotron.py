"""NVIDIA Nemotron 3 TITO tokenizer."""

from __future__ import annotations

from typing import Any

from miles.utils.chat_template_utils.tito_tokenizer.base import FixedTemplate
from miles.utils.chat_template_utils.tito_tokenizer.qwen import Qwen3TITOTokenizer


class Nemotron3TITOTokenizer(Qwen3TITOTokenizer):
    """NVIDIA Nemotron 3 family: ``<|im_end|>\\n`` message boundaries.

    Inherits Qwen3's boundary handling — Nemotron 3 emits the same
    ``<|im_end|>\\n`` after every message and the model stops at
    ``<|im_end|>`` without the trailing newline.

    No fixed jinja is shipped — HF native template is append-only when
    ``truncate_history_thinking=False``.  Multi-user-turn surfaces
    auto-merge that kwarg via ``extra_kwargs`` below; ``{tool}``-only does
    not need it (no user-turn boundary to truncate across).

    The plain-text assistant turn does not roundtrip cleanly under
    sglang's upstream ``nemotron_3`` reasoning parser (it keeps a trailing
    ``\\n`` in ``reasoning_content``), so step-4 ``assistant_text`` soft
    assertion is expected to fail until the parser is patched upstream —
    out of scope for this family registration.
    """

    reasoning_parser = "nemotron_3"
    tool_call_parser = "qwen3_coder"

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"truncate_history_thinking": False},
        consistant_kwargs=["low_effort"],
    )

    _default_assistant_start_str: str = "<|im_start|>assistant\n"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
        )
