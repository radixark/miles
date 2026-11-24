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

# Trailing end-of-answer / end-of-turn markers stripped when special tokens are kept.
_ANSWER_END_TOKENS: dict[str, tuple[str, ...]] = {
    "kimi_k3": ("<|close|>message<|sep|>", "<|close|>response<|sep|>"),
    "harmony": ("<|return|>", "<|end|>"),
    "inkling": ("<|end_message|>",),
    "muse_spark": ("<|eot|>", "<|eom|>"),
    "gemma4": ("<turn|>",),
    "think": ("<|im_end|>", "<｜end▁of▁sentence｜>", "[e~["),
}

_INKLING_MODEL_ROLE = "<|message_model|>"
_INKLING_NON_MODEL_ROLES = ("<|message_tool|>", "<|message_user|>", "<|message_system|>")
_INKLING_END_SAMPLING = "<|content_model_end_sampling|>"


def _trim_answer_end(text: str, family: str) -> tuple[str, bool]:
    """Strip trailing end-of-answer markers for *family*, returning ``(trimmed, was_closed)``."""
    end_tokens = _ANSWER_END_TOKENS.get(family)
    if not end_tokens:
        return text, False
    stripped = text.rstrip()
    trimmed = False
    changed = True
    while changed:
        changed = False
        for token in end_tokens:
            if stripped.endswith(token):
                stripped = stripped[: -len(token)].rstrip()
                trimmed = True
                changed = True
                break
    return (stripped, True) if trimmed else (text, False)


def _extract_inkling_answer(text: str) -> tuple[int, str] | None:
    """Extract visible model text blocks from the final Inkling turn.

    Only ``<|content_text|>`` opening a model message counts (at the start of a
    response after the prompt's ``<|message_model|>``, or immediately after
    ``<|message_model|>``), so tool/user messages never match as a model answer
    and later thinking blocks cannot bleed into the extracted text.

    Returns ``(extent, answer)`` where *extent* is the end of the turn's closed
    Inkling frame (or the last text block start if unclosed).
    """
    content_thinking, content_text = REASONING_DELIMITERS["inkling"]
    model_text = _INKLING_MODEL_ROLE + content_text
    model_thinking = _INKLING_MODEL_ROLE + content_thinking
    end_message = _ANSWER_END_TOKENS["inkling"][0]
    turn_end_marker = end_message + _INKLING_END_SAMPLING

    # Advance past any earlier assistant turn closed by <|end_message|><|content_model_end_sampling|>.
    trimmed = text.rstrip()
    search_end = len(trimmed) - len(_INKLING_END_SAMPLING) if trimmed.endswith(_INKLING_END_SAMPLING) else len(text)
    prev_turn = text.rfind(turn_end_marker, 0, search_end)
    turn_start = 0 if prev_turn == -1 else prev_turn + len(turn_end_marker)

    # Also advance past any validly framed tool/user/system message (ignoring mid-prose mentions).
    for role in _INKLING_NON_MODEL_ROLES:
        limit = len(text)
        while (idx := text.rfind(role, 0, limit)) != -1:
            prefix = text[:idx].rstrip()
            if not prefix or prefix.endswith((end_message, _INKLING_END_SAMPLING)):
                end = text.find(end_message, idx)
                turn_start = max(turn_start, len(text) if end == -1 else end + len(end_message))
                break
            limit = idx

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

    if not parts:
        return None

    close_end = max(
        p + len(end_message) if (p := turn.rfind(end_message)) != -1 else -1,
        p + len(_INKLING_END_SAMPLING) if (p := turn.rfind(_INKLING_END_SAMPLING)) != -1 else -1,
    )
    if close_end != -1 and turn[close_end:].lstrip().startswith(model_thinking):
        extent = len(text)
    elif close_end != -1:
        extent = max(last_pos, turn_start + close_end)
    else:
        extent = last_pos
    # Match InklingResponseParser (inkling_parser.py) which joins visible text blocks with "".
    return (extent, "".join(parts))


def extract_after_reasoning(text: str | None) -> str | None:
    """Return the final answer segment after reasoning, or None if no delimiter matches."""
    if not text:
        return None

    best_extent = -1
    best_answer: str | None = None

    inkling = _extract_inkling_answer(text)
    if inkling is not None:
        best_extent, best_answer = inkling

    # Compare against the rightmost structural boundary: a closed framed block extends to its
    # closing marker so a literal </think> quoted inside it cannot overwrite the answer, while
    # a foreign delimiter quoted inside <think>...</think> stays strictly left of </think>.
    for family, (_, delimiter) in REASONING_DELIMITERS.items():
        if family in ("inkling", "legacy_response"):
            continue
        pos = text.rfind(delimiter)
        if pos != -1 and pos >= best_extent:
            answer, closed = _trim_answer_end(text[pos + len(delimiter) :], family)
            best_extent = len(text) if closed else pos
            best_answer = answer

    if best_answer is not None:
        return best_answer

    _, legacy = REASONING_DELIMITERS["legacy_response"]
    if legacy in text:
        return text.rsplit(legacy, 1)[-1]
    return None
