"""Chat template utilities for agentic-workflow token consistency."""

# Import order matters: message_matcher_hub is the dependency-free base,
# template re-exports its two moved names, and tito_tokenizer builds on both.
from miles.utils.chat_template_utils.message_matcher_hub import (
    assert_messages_append_only_with_allowed_role,
    strict_message_matches,
)
from miles.utils.chat_template_utils.template import (
    apply_chat_template,
    apply_chat_template_from_str,
    extract_tool_dicts,
    load_hf_chat_template,
    normalize_tool_arguments,
)
from miles.utils.chat_template_utils.tito_tokenizer import (
    TEMPLATE_DIR,
    TITOTokenizer,
    TITOTokenizerType,
    get_tito_tokenizer,
    resolve_fixed_chat_template,
    resolve_reasoning_and_tool_call_parser,
)
from miles.utils.chat_template_utils.token_seq_comparator import Mismatch, MismatchType, TokenSeqComparator

__all__ = [
    "TITOTokenizer",
    "TITOTokenizerType",
    "get_tito_tokenizer",
    "TEMPLATE_DIR",
    "resolve_fixed_chat_template",
    "resolve_reasoning_and_tool_call_parser",
    "load_hf_chat_template",
    "apply_chat_template",
    "apply_chat_template_from_str",
    "assert_messages_append_only_with_allowed_role",
    "strict_message_matches",
    "extract_tool_dicts",
    "normalize_tool_arguments",
    "Mismatch",
    "TokenSeqComparator",
    "MismatchType",
]
