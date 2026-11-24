"""Canonical reasoning-to-answer delimiters shared by Session Server (TITO) and RM Hub.

Kept dependency-free so both ``miles.utils.chat_template_utils`` and
``miles.rollout.rm_hub`` share one source of truth without pulling SGLang,
tokenizers, or math-grading libraries across layers.
"""

from __future__ import annotations

# Ordered from model-specific special tokens to generic/legacy markers so an
# explicit delimiter wins over `</think>` or `###Response` in reasoning text.
# Maps family name -> (reasoning_open_token, answer_delimiter).
REASONING_DELIMITERS: dict[str, tuple[str | None, str]] = {
    "kimi_k3": ("<|open|>think<|sep|>", "<|open|>response<|sep|>"),
    "harmony": ("<|channel|>analysis<|message|>", "<|channel|>final<|message|>"),
    "inkling": ("<|content_thinking|>", "<|content_text|>"),
    "muse_spark": ("to=self<|message|>", "to=user<|message|>"),
    "gemma4": ("<|channel>", "<channel|>"),
    "kimi_k2_thinking": ("◁think▷", "◁/think▷"),
    "think": ("<think>", "</think>"),
    "legacy_response": (None, "###Response"),
}


def extract_after_reasoning(text: str | None) -> str | None:
    """Return text after the last occurrence of the first matching answer delimiter, or None if absent."""
    if not text:
        return None
    for _, delimiter in REASONING_DELIMITERS.values():
        if delimiter in text:
            return text.rsplit(delimiter, 1)[-1]
    return None
