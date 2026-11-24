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

# Turn separators that start a new turn in multi-turn framed traces.
_TURN_BOUNDARY_TOKENS: dict[str, str] = {
    "harmony": "<|return|><|start|>",
    "muse_spark": "<|eot|><|start|>",
}

# Message transitions that end an answer block before a subsequent block in the same turn.
_ANSWER_NEXT_BLOCK_TOKENS: dict[str, tuple[str, ...]] = {
    "kimi_k3": ("<|close|>response<|sep|><|open|>", "<|close|>response<|sep|><|close|>message<|sep|><|open|>"),
    "harmony": ("<|end|><|start|>", "<|return|><|start|>"),
    "muse_spark": ("<|eom|><|start|>", "<|eot|><|start|>"),
}

# Trailing end-of-answer / end-of-turn markers stripped when special tokens are kept.
_ANSWER_END_TOKENS: dict[str, tuple[str, ...]] = {
    "kimi_k3": ("<|close|>message<|sep|>", "<|close|>response<|sep|>"),
    "harmony": ("<|return|>", "<|end|>"),
    "inkling": ("<|content_model_end_sampling|>", "<|end_message|>"),
    "muse_spark": ("<|eot|>", "<|eom|>"),
    "gemma4": ("<turn|>",),
    "think": ("<|im_end|>", "<｜end▁of▁sentence｜>", "[e~["),
}

_INKLING_MODEL_ROLE = "<|message_model|>"
_INKLING_NON_MODEL_ROLES = ("<|message_tool|>", "<|message_user|>", "<|message_system|>")
_INKLING_END_MESSAGE = "<|end_message|>"
_INKLING_END_SAMPLING = "<|content_model_end_sampling|>"
_INKLING_TOOL_INVOKE = "<|content_invoke_tool_json|>"
_INKLING_FRAME_PREFIXES = (
    REASONING_DELIMITERS["inkling"][0],
    REASONING_DELIMITERS["inkling"][1],
    _INKLING_MODEL_ROLE,
    *_INKLING_NON_MODEL_ROLES,
)

# Disjoint leading wire prefixes that unambiguously identify a response's own format.
_FRAMED_PREFIXES: dict[str, tuple[str, ...]] = {
    "kimi_k3": ("<|open|>",),
    "harmony": ("<|channel|>", "<|start|>assistant<|channel|>", "<|start|>functions."),
    "inkling": _INKLING_FRAME_PREFIXES,
    "muse_spark": ("<|start|>assistant to=", "<|start|>tool "),
    "gemma4": ("<|channel>", "<channel|>"),
    "kimi_vl": ("◁think▷",),
    "think": ("<think>",),
}


def _detect_framed_family(text: str) -> str | None:
    stripped = text.lstrip()
    if not stripped:
        return None
    for family, prefixes in _FRAMED_PREFIXES.items():
        if stripped.startswith(prefixes):
            return family
    first_token = stripped.split(None, 1)[0]
    if _INKLING_TOOL_INVOKE in first_token:
        return "inkling"
    if stripped.startswith("to=") and "<|message|>" in first_token:
        return "muse_spark"
    return None


def _trim_answer_end(text: str, family: str) -> str:
    """Truncate before any subsequent framed block, then strip trailing end-of-answer markers."""
    next_tokens = _ANSWER_NEXT_BLOCK_TOKENS.get(family)
    if next_tokens:
        positions = [pos for token in next_tokens if (pos := text.find(token)) != -1]
        if positions:
            text = text[: min(positions)]

    end_tokens = _ANSWER_END_TOKENS.get(family)
    if not end_tokens:
        return text
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
    return stripped if trimmed else text


def _extract_inkling_answer(text: str) -> str | None:
    """Extract visible model text blocks from the final Inkling turn.

    Only ``<|content_text|>`` opening a model message counts (at the start of a
    response after the prompt's ``<|message_model|>``, or immediately after
    ``<|message_model|>``), so tool/user messages never match as a model answer
    and later thinking blocks cannot bleed into the extracted text.
    """
    _, content_text = REASONING_DELIMITERS["inkling"]
    model_text = _INKLING_MODEL_ROLE + content_text
    turn_end_marker = _INKLING_END_MESSAGE + _INKLING_END_SAMPLING

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
            if not prefix or prefix.endswith((_INKLING_END_MESSAGE, _INKLING_END_SAMPLING)):
                end = text.find(_INKLING_END_MESSAGE, idx)
                turn_start = max(turn_start, len(text) if end == -1 else end + len(_INKLING_END_MESSAGE))
                break
            limit = idx

    turn = text[turn_start:]
    parts: list[str] = []
    cursor = 0
    leading = turn.lstrip() if turn_start == 0 else ""
    if turn_start == 0 and leading.startswith(content_text):
        body_start = len(turn) - len(leading) + len(content_text)
        body_end = turn.find(_INKLING_END_MESSAGE, body_start)
        parts.append(turn[body_start:] if body_end == -1 else turn[body_start:body_end])
        cursor = len(turn) if body_end == -1 else body_end + len(_INKLING_END_MESSAGE)

    while (pos := turn.find(model_text, cursor)) != -1:
        body_start = pos + len(model_text)
        body_end = turn.find(_INKLING_END_MESSAGE, body_start)
        parts.append(turn[body_start:] if body_end == -1 else turn[body_start:body_end])
        cursor = len(turn) if body_end == -1 else body_end + len(_INKLING_END_MESSAGE)

    # Match InklingResponseParser (inkling_parser.py) which joins visible text blocks with "".
    return "".join(parts) if parts else None


def extract_after_reasoning(text: str | None) -> str | None:
    """Return the final answer segment after reasoning, or None if no delimiter matches."""
    if not text:
        return None

    # When the response opens with a known wire format, parse only with that family's rules
    # so foreign markers quoted in reasoning/answers never hijack extraction and an unclosed
    # final turn returns None instead of matching an earlier turn or foreign token.
    framed_family = _detect_framed_family(text)
    if framed_family == "inkling":
        return _extract_inkling_answer(text)
    if framed_family is not None:
        turn = text
        turn_boundary = _TURN_BOUNDARY_TOKENS.get(framed_family)
        if turn_boundary and (boundary_pos := text.rfind(turn_boundary)) != -1:
            turn = text[boundary_pos + len(turn_boundary) :]
        _, delimiter = REASONING_DELIMITERS[framed_family]
        pos = turn.rfind(delimiter)
        if pos == -1:
            return None
        return _trim_answer_end(turn[pos + len(delimiter) :], framed_family)

    # Unframed / prompt-opened completions (e.g. `<think>` emitted in the prompt): pick the
    # rightmost structural delimiter, falling back to `###Response` only when none matches.
    best_pos = -1
    best_answer: str | None = None
    for family, (_, delimiter) in REASONING_DELIMITERS.items():
        if family == "legacy_response":
            continue
        marker = _INKLING_MODEL_ROLE + delimiter if family == "inkling" else delimiter
        pos = text.rfind(marker)
        if pos > best_pos:
            best_pos = pos
            best_answer = _trim_answer_end(text[pos + len(marker) :], family)

    if best_answer is not None:
        return best_answer

    _, legacy = REASONING_DELIMITERS["legacy_response"]
    if legacy in text:
        return text.rsplit(legacy, 1)[-1]
    return None
