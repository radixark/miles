"""Muse Spark TITO tokenizer."""

from __future__ import annotations

from typing import Any

from miles.utils.chat_template_utils.tito_tokenizer.base import FixedTemplate, TITOTokenizer
from miles.utils.chat_template_utils.token_seq_comparator import TokenSeqComparator


class MuseSparkTITOTokenizer(TITOTokenizer):
    FIXED_TEMPLATE = FixedTemplate(template="muse_spark_fixed.jinja")
    reasoning_parser = "muse"
    tool_call_parser = "muse"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        # Keep the recipient header and body in one comparator segment; the
        # message separator is internal to a turn, not a role boundary.
        super().__init__(
            tokenizer,
            chat_template_kwargs=chat_template_kwargs,
            assistant_start_str=assistant_start_str or "<|start|>assistant",
            special_token_ids=TokenSeqComparator.collect_special_ids(tokenizer)
            - {tokenizer.convert_tokens_to_ids("<|message|>")},
        )
