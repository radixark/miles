"""Chat template utilities for agentic-workflow token consistency."""

from miles.utils.chat_template_utils.autofix import TEMPLATE_DIR, try_get_fixed_chat_template
from miles.utils.chat_template_utils.template import (
    apply_chat_template_from_str,
    extract_tool_dicts,
    load_hf_chat_template,
)

__all__ = [
    "TEMPLATE_DIR",
    "try_get_fixed_chat_template",
    "load_hf_chat_template",
    "apply_chat_template_from_str",
    "extract_tool_dicts",
]
