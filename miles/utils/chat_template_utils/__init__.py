"""Chat template utilities for agentic-workflow token consistency."""

from miles.utils.chat_template_utils.autofix import TEMPLATE_DIR, try_get_fixed_chat_template
from miles.utils.chat_template_utils.template import (
    apply_chat_template,
    apply_chat_template_from_str,
    assert_messages_append_only,
    extract_tool_dicts,
    load_hf_chat_template,
    message_matches,
)
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, TITOTokenizerType, get_tito_tokenizer
from miles.utils.chat_template_utils.token_seq_comparator import Mismatch, MismatchType, Segment, TokenSeqComparator

__all__ = [
    "TITOTokenizer",
    "TITOTokenizerType",
    "get_tito_tokenizer",
    "TEMPLATE_DIR",
    "try_get_fixed_chat_template",
    "load_hf_chat_template",
    "apply_chat_template",
    "apply_chat_template_from_str",
    "assert_messages_append_only",
    "message_matches",
    "extract_tool_dicts",
    "Mismatch",
    "TokenSeqComparator",
    "MismatchType",
    "Segment",
]
