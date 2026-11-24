"""Canonical reasoning-to-answer delimiters shared by Session Server (TITO) and RM Hub.

Kept stdlib-only so both ``miles.utils.chat_template_utils`` and
``miles.rollout.rm_hub`` share one source of truth without pulling SGLang,
tokenizers, or math-grading libraries across layers.
"""

from __future__ import annotations

import re

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

_INKLING_MODEL_ROLE = "<|message_model|>"
# Inkling turns end at `<|end_message|><|content_model_end_sampling|>` or framed tool/user/system messages.
_INKLING_TURN_BREAK_RE = re.compile(
    r"<\|end_message\|><\|content_model_end_sampling\|>(?=\s*\S)"
    r"|(?:^|(?<=<\|end_message\|>)|(?<=<\|content_model_end_sampling\|>))"
    r"\s*<\|message_(?:tool|user|system)\|>(?:.*?<\|end_message\|>|$)",
    re.DOTALL,
)
_INKLING_TEXT_BLOCK_RE = re.compile(r"<\|message_model\|><\|content_text\|>(.*?)(?:<\|end_message\|>|$)", re.DOTALL)

# Turn separators that bound multi-turn traces so an unclosed final turn cannot match an earlier answer.
_TURN_BOUNDARY_TOKENS: dict[str, tuple[str, ...]] = {
    "harmony": ("<|return|><|start|>", "<|end|><|start|>"),
    "muse_spark": ("<|eot|><|start|>",),
}
# Same-turn transitions that end a framed answer block before a subsequent block.
_NEXT_BLOCK_RE: dict[str, re.Pattern[str]] = {
    "kimi_k3": re.compile(r"<\|close\|>response<\|sep\|>(?:<\|close\|>message<\|sep\|>)?<\|open\|>"),
    "muse_spark": re.compile(r"<\|eom\|><\|start\|>"),
}
# Trailing end-of-answer / end-of-turn markers stripped in linear time when rollout_skip_special_tokens=False.
_TRAILING_END_TOKENS: dict[str, tuple[str, ...]] = {
    "kimi_k3": ("<|close|>response<|sep|>", "<|close|>message<|sep|>"),
    "harmony": ("<|return|>", "<|end|>"),
    "muse_spark": ("<|eot|>", "<|eom|>"),
    "gemma4": ("<turn|>",),
    "think": ("<|im_end|>", "<｜end▁of▁sentence｜>", "[e~["),
}


def _trim_answer_end(text: str, family: str) -> str:
    if (pattern := _NEXT_BLOCK_RE.get(family)) and (match := pattern.search(text)):
        text = text[: match.start()]
    if not (end_tokens := _TRAILING_END_TOKENS.get(family)):
        return text
    trimmed, stripped = text.rstrip(), False
    while trimmed:
        for token in end_tokens:
            if trimmed.endswith(token):
                trimmed, stripped = trimmed[: -len(token)].rstrip(), True
                break
        else:
            break
    return trimmed if stripped else text


def _extract_inkling_answer(text: str) -> tuple[int, str] | None:
    """Extract concatenated model ``<|content_text|>`` blocks from the final Inkling turn."""
    turn_start = max((m.end() for m in _INKLING_TURN_BREAK_RE.finditer(text)), default=0)
    turn = text[turn_start:]
    if turn_start == 0 and (leading := turn.lstrip()).startswith(REASONING_DELIMITERS["inkling"][1]):
        turn = _INKLING_MODEL_ROLE + leading
    matches = list(_INKLING_TEXT_BLOCK_RE.finditer(turn))
    if not matches:
        return None
    shift = len(turn) - len(text[turn_start:])
    return (max(0, turn_start + matches[-1].start() - shift), "".join(m.group(1) for m in matches))


def extract_after_reasoning(text: str | None) -> str | None:
    """Return the final answer segment after reasoning, or None if no delimiter matches."""
    if not text:
        return None

    best_pos, best_answer = -1, None
    if (inkling := _extract_inkling_answer(text)) is not None:
        best_pos, best_answer = inkling

    for family, (_, delimiter) in REASONING_DELIMITERS.items():
        if family in ("inkling", "legacy_response"):
            continue
        turn_start = 0
        for boundary in _TURN_BOUNDARY_TOKENS.get(family, ()):
            if (boundary_pos := text.rfind(boundary)) != -1:
                turn_start = max(turn_start, boundary_pos + len(boundary))
        if (pos := text.rfind(delimiter, turn_start)) > best_pos:
            best_pos, best_answer = pos, _trim_answer_end(text[pos + len(delimiter) :], family)

    if best_answer is not None:
        return best_answer

    _, legacy = REASONING_DELIMITERS["legacy_response"]
    return text.rsplit(legacy, 1)[-1] if legacy in text else None
