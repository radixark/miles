"""Canonical reasoning-to-answer delimiters shared by Session Server (TITO) and RM Hub.

Kept dependency-free so both ``miles.utils.chat_template_utils`` and
``miles.rollout.rm_hub`` share one source of truth without pulling SGLang,
tokenizers, or math-grading libraries across layers.
"""

from __future__ import annotations

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

KIMI_K3_ANSWER_DELIMITER = "<|open|>response<|sep|>"
HARMONY_ANSWER_DELIMITER = "<|channel|>final<|message|>"
INKLING_ANSWER_DELIMITER = "<|content_text|>"
MUSE_SPARK_ANSWER_DELIMITER = "to=user<|message|>"
GEMMA4_ANSWER_DELIMITER = "<channel|>"
KIMI_K2_THINK_CLOSE = "◁/think▷"
LEGACY_RESPONSE_DELIMITER = "###Response"

# Ordered from model-specific special tokens to generic/legacy markers so an
# explicit delimiter wins over `</think>` or `###Response` in reasoning text.
REASONING_ANSWER_DELIMITERS: tuple[str, ...] = (
    KIMI_K3_ANSWER_DELIMITER,  # Kimi K3
    HARMONY_ANSWER_DELIMITER,  # GPT-OSS (Harmony)
    INKLING_ANSWER_DELIMITER,  # Inkling / Inkling 2
    MUSE_SPARK_ANSWER_DELIMITER,  # Muse Spark
    GEMMA4_ANSWER_DELIMITER,  # Gemma-4
    KIMI_K2_THINK_CLOSE,  # Kimi K2-Thinking
    THINK_CLOSE,  # Qwen3, DeepSeek, GLM, MiniMax, Nemotron, MiMo, Kimi K2.5+, Step
    LEGACY_RESPONSE_DELIMITER,  # DeepScaleR legacy template
)


def extract_after_reasoning(text: str | None) -> str | None:
    """Return text after the last occurrence of the first matching delimiter, or None if absent."""
    if not text:
        return None
    for delimiter in REASONING_ANSWER_DELIMITERS:
        if delimiter in text:
            return text.rsplit(delimiter, 1)[-1]
    return None
