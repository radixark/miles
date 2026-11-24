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
_TURN_BOUNDARY_TOKENS: dict[str, tuple[str, ...]] = {
    "harmony": ("<|return|><|start|>", "<|end|><|start|>user", "<|end|><|start|>developer", "<|end|><|start|>system"),
    "muse_spark": ("<|eot|><|start|>",),
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
    _INKLING_END_SAMPLING,
)

# Disjoint leading wire prefixes that unambiguously identify a response's own format.
_FRAMED_PREFIXES: dict[str, tuple[str, ...]] = {
    "kimi_k3": ("<|open|>",),
    "harmony": ("<|channel|>", "<|start|>assistant<|channel|>", "<|start|>functions."),
    "inkling": _INKLING_FRAME_PREFIXES,
    "muse_spark": ("<|start|>tool ",),
    "gemma4": ("<|channel>", "<channel|>"),
    "kimi_vl": ("◁think▷",),
    "think": ("<think>",),
}


def _is_closed_after_think(text: str, open_marker: str, close_token: str) -> bool:
    """Return True when *open_marker* is closed by *close_token* after any literal ``</think>``."""
    pos = text.rfind(open_marker)
    if pos == -1:
        return False
    tail = text[pos + len(open_marker) :]
    think_close = REASONING_DELIMITERS["think"][1]
    if think_close not in tail:
        return True
    close_pos = tail.find(close_token)
    return close_pos != -1 and close_pos > tail.rfind(think_close)


def _detect_framed_family(text: str) -> str | None:
    """Identify the response's wire format from leading or trailing special tokens."""
    stripped = text.lstrip()
    if not stripped:
        return None
    for family, prefixes in _FRAMED_PREFIXES.items():
        if stripped.startswith(prefixes):
            return family

    # Both Harmony and Muse Spark use `<|start|>assistant to=...` / `to=...`; Harmony attaches
    # `<|channel|>` directly to the recipient before `<|message|>`, while Muse Spark never does.
    if stripped.startswith("<|start|>assistant to="):
        header = stripped[len("<|start|>assistant ") :].split(None, 1)[0].split("<|message|>", 1)[0]
        return "harmony" if "<|channel|>" in header else "muse_spark"

    if stripped.startswith(("<|start|>user", "<|start|>developer", "<|start|>system")):
        harmony_end = text.find("<|end|><|start|>")
        muse_end = text.find("<|eot|><|start|>")
        if harmony_end != -1 and (muse_end == -1 or harmony_end < muse_end):
            return "harmony"
        if muse_end != -1:
            return "muse_spark"

    first_token = stripped.split(None, 1)[0]
    if stripped.startswith("to="):
        header = first_token.split("<|message|>", 1)[0]
        if "<|channel|>" in header:
            return "harmony"
        if "<|message|>" in first_token:
            return "muse_spark"

    # Inkling direct tool call right after prompt's `<|message_model|>`: `<recipient><|content_invoke_tool_json|>{...}`.
    # Native InklingResponseParser accepts empty or `_` recipients and length-cut JSON before `<|end_message|>`.
    if _INKLING_TOOL_INVOKE in first_token:
        recipient, _ = first_token.split(_INKLING_TOOL_INVOKE, 1)
        after_invoke = stripped[len(recipient) + len(_INKLING_TOOL_INVOKE) :].lstrip()
        if all(ch.isalnum() or ch in "_-." for ch in recipient) and after_invoke.startswith("{"):
            return "inkling"

    # Prompt-opened completions omit their opening token; identify by trailing closing special tokens.
    rstripped = text.rstrip()
    if rstripped.endswith("<|close|>think<|sep|>") or (
        rstripped.endswith(_ANSWER_END_TOKENS["kimi_k3"])
        and _is_closed_after_think(text, REASONING_DELIMITERS["kimi_k3"][1], "<|close|>response<|sep|>")
    ):
        return "kimi_k3"
    if rstripped.endswith(_ANSWER_END_TOKENS["gemma4"]) and _is_closed_after_think(
        text, REASONING_DELIMITERS["gemma4"][1], "<turn|>"
    ):
        return "gemma4"
    if rstripped.endswith(_ANSWER_END_TOKENS["think"]) and REASONING_DELIMITERS["think"][1] in text:
        return "think"

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

    # When special tokens identify the wire format, parse only with that family's rules so
    # foreign markers quoted in reasoning/answers never hijack extraction and an unclosed
    # final turn returns None instead of matching an earlier turn or foreign token.
    framed_family = _detect_framed_family(text)
    if framed_family == "inkling":
        return _extract_inkling_answer(text)
    if framed_family is not None:
        turn = text
        turn_boundaries = _TURN_BOUNDARY_TOKENS.get(framed_family)
        if turn_boundaries:
            turn_start = 0
            for boundary in turn_boundaries:
                if (boundary_pos := text.rfind(boundary)) != -1:
                    turn_start = max(turn_start, boundary_pos + len(boundary))
            turn = text[turn_start:]
        _, delimiter = REASONING_DELIMITERS[framed_family]
        pos = turn.rfind(delimiter)
        if pos == -1:
            return None
        return _trim_answer_end(turn[pos + len(delimiter) :], framed_family)

    # Unframed / prompt-opened completions without trailing end tokens: pick the rightmost
    # structural delimiter, falling back to `###Response` only when none matches.
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
