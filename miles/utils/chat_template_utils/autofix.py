"""Auto-fix chat templates for agentic-workflow token consistency.

Some model families ship chat templates that use ``loop.last`` or similar
context-dependent Jinja logic, which breaks the append-only invariant
required by sglang's pretokenized prefix mechanism.  This module maps
HuggingFace checkpoint names to fixed templates bundled in the same directory.
"""

import fnmatch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

# (glob pattern, fixed template filename) — checked in order, first match wins.
# Patterns are matched case-insensitively against the hf_checkpoint string.
_FIX_RULES: list[tuple[str, str]] = [
    ("*qwen3.5*", "qwen3.5_fixed.jinja"),
    ("*qwen3-next-*-thinking*", "qwen3_thinking_2507_and_next_fixed.jinja"),
    ("*qwen3-*b-thinking-2507*", "qwen3_thinking_2507_and_next_fixed.jinja"),
    ("*qwen3-[0-9]*", "qwen3_fixed.jinja"),
]


def try_get_fixed_chat_template(hf_checkpoint: str) -> str | None:
    """Return the path to a fixed chat template for *hf_checkpoint*, or ``None``.

    Matches the checkpoint name against known glob patterns (case-insensitive).
    Returns the absolute path to the bundled ``.jinja`` file if a rule matches,
    otherwise ``None``.
    """
    name = hf_checkpoint.lower()
    for pattern, filename in _FIX_RULES:
        if fnmatch.fnmatch(name, pattern):
            path = str(TEMPLATE_DIR / filename)
            logger.info("Checkpoint %r matched %r -> using fixed chat template %s", hf_checkpoint, pattern, path)
            return path
    return None
