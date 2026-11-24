"""Canonical reasoning-to-answer delimiters shared by Session Server (TITO) and RM Hub.

Kept dependency-free so both ``miles.utils.chat_template_utils`` and
``miles.rollout.rm_hub`` share one source of truth without pulling SGLang,
tokenizers, or math-grading libraries across layers.
"""

from __future__ import annotations

# Maps family name -> (reasoning_open_token, answer_delimiter).
REASONING_DELIMITERS: dict[str, tuple[str | None, str]] = {
    "kimi_k3": ("<|open|>think<|sep|>", "<|open|>response<|sep|>"),
    "harmony": ("<|channel|>analysis<|message|>", "<|channel|>final<|message|>"),
    "inkling": ("<|content_thinking|>", "<|content_text|>"),
    "muse_spark": ("to=self<|message|>", "to=user<|message|>"),
    "gemma4": ("<|channel>", "<channel|>"),
    "kimi_vl": ("◁think▷", "◁/think▷"),
    "think": ("<think>", "</think>"),
    "legacy_response": (None, "###Response"),
}

# Closing markers that terminate an answer block in framed formats.
_ANSWER_END_TOKENS: dict[str, tuple[str, ...]] = {
    "kimi_k3": ("<|close|>response<|sep|>",),
    "harmony": ("<|return|>", "<|end|>"),
    "inkling": ("<|end_message|>",),
    "muse_spark": ("<|eot|>", "<|eom|>"),
    "gemma4": ("<turn|>",),
}

_INKLING_MODEL_ROLE = "<|message_model|>"
_INKLING_NON_MODEL_ROLES = ("<|message_tool|>", "<|message_user|>", "<|message_system|>")


def _trim_answer_end(text: str, family: str) -> str:
    end_tokens = _ANSWER_END_TOKENS.get(family)
    if not end_tokens:
        return text
    positions = [pos for token in end_tokens if (pos := text.find(token)) != -1]
    return text[: min(positions)] if positions else text


def _extract_inkling_answer(text: str) -> tuple[int, str] | None:
    """Extract visible model text blocks from the final Inkling turn.

    Only ``<|content_text|>`` opening a model message counts (at the start of a
    response after the prompt's ``<|message_model|>``, or immediately after
    ``<|message_model|>``), so tool/user messages never match as a model answer
    and later thinking blocks cannot bleed into the extracted text.
    """
    _, content_text = REASONING_DELIMITERS["inkling"]
    model_text = _INKLING_MODEL_ROLE + content_text
    end_message = _ANSWER_END_TOKENS["inkling"][0]

    # Restrict to the final model turn after any interleaved tool/user/system message.
    turn_start = 0
    for role in _INKLING_NON_MODEL_ROLES:
        idx = text.rfind(role)
        if idx != -1:
            end = text.find(end_message, idx)
            turn_start = max(turn_start, len(text) if end == -1 else end + len(end_message))

    turn = text[turn_start:]
    parts: list[str] = []
    last_pos = -1
    cursor = 0
    if turn_start == 0 and turn.startswith(content_text):
        last_pos = 0
        body_start = len(content_text)
        body_end = turn.find(end_message, body_start)
        parts.append(turn[body_start:] if body_end == -1 else turn[body_start:body_end])
        cursor = len(turn) if body_end == -1 else body_end + len(end_message)

    while (pos := turn.find(model_text, cursor)) != -1:
        last_pos = turn_start + pos
        body_start = pos + len(model_text)
        body_end = turn.find(end_message, body_start)
        parts.append(turn[body_start:] if body_end == -1 else turn[body_start:body_end])
        cursor = len(turn) if body_end == -1 else body_end + len(end_message)

    return (last_pos, "\n".join(parts)) if parts else None


def extract_after_reasoning(text: str | None) -> str | None:
    """Return the final answer segment after reasoning, or None if no delimiter matches."""
    if not text:
        return None

    best_pos = -1
    best_answer: str | None = None

    inkling = _extract_inkling_answer(text)
    if inkling is not None:
        best_pos, best_answer = inkling

    # Pick the rightmost structural delimiter so a marker quoted inside reasoning
    # never shadows the model's real closing delimiter that follows it.
    for family, (_, delimiter) in REASONING_DELIMITERS.items():
        if family in ("inkling", "legacy_response"):
            continue
        pos = text.rfind(delimiter)
        if pos > best_pos:
            best_pos = pos
            best_answer = _trim_answer_end(text[pos + len(delimiter) :], family)

    if best_answer is not None:
        return best_answer

    _, legacy = REASONING_DELIMITERS["legacy_response"]
    if legacy in text:
        return text.rsplit(legacy, 1)[-1]
    return None
