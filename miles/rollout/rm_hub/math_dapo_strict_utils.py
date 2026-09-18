"""Opt-in DAPO grader for a single integer on a final, standalone Answer line.

Unlike the historical DAPO normalizer, this parser never searches for a convenient
number inside an expression or discards explanatory suffixes. It is not a general
MATH/LaTeX equivalence grader. Text-only callers cannot infer generation status;
the RM dispatcher must pass ``is_complete=False`` for unfinished samples.
"""

import re

_MARKDOWN = ("**", "__", "*", "_", "`")
_MARKDOWN_WRAPPERS = tuple((marker, marker) for marker in _MARKDOWN)
_WRAPPERS = _MARKDOWN_WRAPPERS + (
    (r"\(", r"\)"),
    (r"\[", r"\]"),
    ("$$", "$$"),
    ("$", "$"),
    (r"\boxed{", "}"),
)
_EOS = ("<｜end▁of▁sentence｜>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>")
_INTEGER = re.compile(r"[+-]?[0-9]+")
_LABEL_INTEGER = re.compile(r"[+-]?[0-9]+(?:\.0+)?")
_ANSWER_PREFIXES = (r"Answer[ \t]*:",) + tuple(
    rf"{re.escape(marker)}Answer(?:[ \t]*:{re.escape(marker)}|{re.escape(marker)}[ \t]*:)"
    for marker in _MARKDOWN
)
_ANSWER_LINE = re.compile(r"(?:" + "|".join(_ANSWER_PREFIXES) + r")[ \t]*(.*)", re.IGNORECASE)
_HEADING = re.compile(r"#{1,6}[ \t]+")
_FENCE = re.compile(r"(`{3,}|~{3,})")


def _canonical_integer(value: str) -> str:
    # String arithmetic avoids both float rounding and Python's int digit limit.
    digits = value.lstrip("+-").lstrip("0") or "0"
    return "-" + digits if value.startswith("-") and digits != "0" else digits


def _normalize_label(ground_truth: str | int | float) -> str:
    """Accept exact integer labels (including .0 suffixes), never truncate floats.

    Float labels are accepted only when integral and inside the safe-integer
    range. Larger labels must arrive as strings/ints so precision is not already
    lost before grading. Invalid labels are dataset errors, not model mistakes.
    """
    if isinstance(ground_truth, bool):
        raise ValueError("dapo_strict requires an integer label, not bool")
    if isinstance(ground_truth, float):
        if not ground_truth.is_integer() or abs(ground_truth) > 2**53 - 1:
            raise ValueError("dapo_strict requires an exact integer label; use str/int for large values")
        ground_truth = int(ground_truth)
    if not isinstance(ground_truth, (str, int)):
        raise ValueError("dapo_strict requires an integer label as str/int or a safe integral float")
    label = str(ground_truth).strip()
    if _LABEL_INTEGER.fullmatch(label) is None:
        raise ValueError(f"dapo_strict requires an integer label, got {ground_truth!r}")
    return _canonical_integer(label.split(".", 1)[0])


def _unwrap_once(text: str, wrappers: tuple[tuple[str, str], ...]) -> str | None:
    for left, right in wrappers:
        if len(text) >= len(left) + len(right) and text.startswith(left) and text.endswith(right):
            return text[len(left) : -len(right)].strip()
    return None


def _parse_integer(text: str) -> str | None:
    while True:
        if _INTEGER.fullmatch(text):
            return _canonical_integer(text)
        unwrapped = _unwrap_once(text, _WRAPPERS)
        if unwrapped is None:
            return None
        text = unwrapped


def _answer_payloads(line: str) -> tuple[str, ...]:
    # Try the original line before peeling whole-line emphasis: e.g.
    # **Answer:** **45** has separate label/value wrappers, not one outer pair.
    line = _HEADING.sub("", line, count=1)
    payloads = []
    while True:
        match = _ANSWER_LINE.fullmatch(line)
        if match:
            payloads.append(match[1].strip())
        unwrapped = _unwrap_once(line, _MARKDOWN_WRAPPERS)
        if unwrapped is None:
            return tuple(payloads)
        line = unwrapped


def _final_section(text: str) -> str | None:
    text = text.strip()
    while True:
        eos = next((token for token in _EOS if text.endswith(token)), None)
        if eos is None:
            break
        text = text[: -len(eos)].rstrip()
    if any(token in text for token in _EOS):
        return None  # An EOS inside the response is not harmless suffix padding.

    # A lone closing tag is normal when the chat template prefills <think>.
    # Never grade a temporary answer inside unfinished or malformed reasoning.
    opens, closes = text.count("<think>"), text.count("</think>")
    if opens > 1 or closes > 1 or (opens and not closes):
        return None
    if closes:
        before, after = text.split("</think>")
        if "<think>" in after or not after.strip():
            return None
        # With an explicit opener, preserve any preceding content so competing
        # Answer lines outside the reasoning block cannot be silently discarded.
        prefix = before.split("<think>", 1)[0] if opens else ""
        text = prefix + "\n" + after
    return text.strip()


def extract_final_integer(solution_str: str) -> str | None:
    """Parse one final Answer line, ignoring only closed reasoning/code blocks.

    Supports complete Markdown emphasis/inline code, math delimiters and boxed
    wrappers. Multiple Answer lines outside reasoning, multiple values, partial
    wrappers, non-integers and trailing prose all fail closed. No tail window is
    used: long integers and earlier competing answers must remain visible.
    """
    text = _final_section(solution_str)
    if not text:
        return None
    lines = text.splitlines()
    answer_lines = []
    fence = None
    for index, line in enumerate(lines):
        line = line.strip()
        marker = _FENCE.match(line)
        if fence is not None:
            if marker and line == marker[1] and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            continue
        if marker:
            fence = marker[1]
            continue
        payloads = _answer_payloads(line)
        if payloads:
            answer_lines.append((index, payloads))
    if fence is not None or len(answer_lines) != 1 or answer_lines[0][0] != len(lines) - 1:
        return None
    for payload in answer_lines[0][1]:
        integer = _parse_integer(payload)
        if integer is not None:
            return integer
    return None


def compute_score(
    solution_str: str, ground_truth: str | int | float, *, is_complete: bool = True
) -> dict[str, float | bool | str | None]:
    """Return DAPO +/-1 reward plus validity, without changing the legacy grader.

    ``valid=False`` means the response has no gradeable final integer (including
    non-completed samples). Its -1 is a format/completion penalty, NOT evidence of
    an incorrect mathematical answer. Correctness is compared only when valid.
    Text-only calls grade the text as provided; the default RM route additionally
    requires Sample.Status.COMPLETED, so truncation cannot earn a spurious +1.
    """
    label = _normalize_label(ground_truth)
    pred = extract_final_integer(solution_str) if is_complete else None
    correct = pred is not None and pred == label
    return {"score": 1.0 if correct else -1.0, "acc": correct, "pred": pred, "valid": pred is not None}
