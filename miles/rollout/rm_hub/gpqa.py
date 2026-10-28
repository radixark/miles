import re
import string
from collections.abc import Iterable

DEFAULT_VALID_LETTERS = list(string.ascii_uppercase[:8])


def _strip_chain_of_thought(text: str) -> str:
    if not text:
        return ""

    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]

    return text


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# Keywords match case-insensitively, but the option itself must be a standalone capital
# letter, so words such as "definitely" or the article "A" are never read as options. At most one
# lowercase filler word may sit between keyword and letter, and never a negation ("is not A").
_LETTER = r"\(?([A-Z])\)?(?![A-Za-z0-9])"
_FILLER = r"(?:(?!(?:not|never|no)\b)[a-z]+\s+)?"
_LETTER_PATTERNS = [
    r"(?i:final\s*(?:answer|option))\b\s*(?i:is)?\s*[:\-]?\s*" + _FILLER + _LETTER,
    r"(?i:answer|option|choice)\b\s*(?i:is)?\s*[:\-]?\s*" + _FILLER + _LETTER,
    r"\b([A-Z])\)?\s*(?i:is\s*(?:the\s*)?correct)",
]
# An option label at the start of the answer, e.g. "C. 42 eV" or "(C) 42 eV" inside a box.
# The text after it must not be another option label, so "(A) (B) (C) (D)" is not a commitment.
_LEADING_LABEL = r"\s*\(?([A-Z])(?:[.:]|\))\s+(?!\(?[A-Z](?:[.:)\s]|$))\S.*"
# A letter that ends the answer, e.g. "I pick B", "Therefore, B." or a bare "(B)".
_TRAILING_LETTER = r"(?:^|[^\w(])\(?([A-Z])\)?[.!]?\s*$"
# Text right before that trailing letter that makes it a hedge or a rejection rather than a
# commitment: one item of a list ("A B C D", "A or B") or a negated option ("not C").
_NOT_COMMITTED_TAIL = r"(?:\b[A-Z]\)?\s*(?:[,./;]|\bor\b|\band\b)?|(?i:\b(?:not|never|no)))\s*\(?$"


def _unwrap_markup(text: str) -> str:
    """Drop markdown emphasis, code ticks, math delimiters and LaTeX wrappers such as `\boxed{C}` or `\text{C}`, keeping their contents."""
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\(?:boxed|text|textbf|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", text)
    return re.sub(r"\\[()\[\]]|[*_`$]", "", text)


def _extract_letter_from_response(response: str, valid_letters: Iterable[str]) -> str | None:
    """
    Extract the single option letter the model committed to, or None if it did not commit.
    """
    if not response:
        return None

    text = _unwrap_markup(_strip_chain_of_thought(response)).strip()
    valid_letters = {letter.upper() for letter in valid_letters}
    for pattern in _LETTER_PATTERNS:
        match = re.search(pattern, text)
        if match and match.group(1) in valid_letters:
            return match.group(1)

    last_line = text.splitlines()[-1] if text else ""
    match = re.search(_TRAILING_LETTER, last_line)
    if match and match.group(1) in valid_letters and not re.search(_NOT_COMMITTED_TAIL, last_line[: match.start(1)]):
        return match.group(1)

    # Only a single-line answer may be read by its leading label; a multi-line walkthrough that
    # opens with "A. Incorrect because ..." has not chosen A.
    match = re.fullmatch(_LEADING_LABEL, text) if "\n" not in text else None
    if match and match.group(1) in valid_letters:
        return match.group(1)

    return None


def _unique_choice_letter(response: str, choices: list, valid_letters: list[str]) -> str | None:
    """Return the letter of the only choice whose text appears in the response, if exactly one does."""
    # Match longer choices first and blank them out, so a choice nested inside the named one
    # ("2-butanol" inside "(R)-2-butanol") is not counted as a second, hedged option.
    remaining = f" {_normalize_text(_strip_chain_of_thought(response))} "
    targets = [
        (f" {_normalize_text(str(choice))} ", letter) for letter, choice in zip(valid_letters, choices, strict=False)
    ]
    matched = []
    for target, letter in sorted(targets, key=lambda item: len(item[0]), reverse=True):
        if target.strip() and target in remaining:
            matched.append(letter)
            remaining = remaining.replace(target, " | ")
    return matched[0] if len(matched) == 1 else None


def compute_gpqa_reward(response: str, label, metadata: dict | None = None) -> float:
    """Rule-based scorer for GPQA-style multiple-choice evaluation."""
    if response is None:
        return 0.0

    metadata = metadata or {}

    choices = metadata.get("choices")
    if isinstance(choices, dict):
        choices = list(choices.values())
    elif choices is not None:
        choices = list(choices)

    valid_letters = metadata.get("valid_letters")
    if valid_letters:
        valid_letters = [str(letter).upper() for letter in valid_letters]
    elif choices:
        valid_letters = list(string.ascii_uppercase[: len(choices)])
    else:
        valid_letters = DEFAULT_VALID_LETTERS

    correct_letter = metadata.get("correct_letter")
    if isinstance(correct_letter, str):
        correct_letter = correct_letter.strip().upper()
    else:
        correct_letter = None

    label_text = None
    if isinstance(label, str):
        label_text = label.strip()
        if len(label_text) == 1 and label_text.upper() in valid_letters and not correct_letter:
            correct_letter = label_text.upper()
    elif isinstance(label, (int, float)):
        idx = int(label)
        if 0 <= idx < len(valid_letters):
            correct_letter = valid_letters[idx]

    if not correct_letter and choices and label_text:
        normalized_label = _normalize_text(label_text)
        for idx, choice in enumerate(choices):
            if _normalize_text(str(choice)) == normalized_label:
                correct_letter = valid_letters[idx]
                metadata.setdefault("correct_answer", choice)
                break

    extracted_letter = _extract_letter_from_response(response, valid_letters)
    if extracted_letter and correct_letter:
        return 1.0 if extracted_letter == correct_letter else 0.0

    # Without a committed letter, credit the answer text only when it names exactly one option.
    # A response that lists several options is a hedge and scores 0.
    if correct_letter and choices:
        return 1.0 if _unique_choice_letter(response, choices, valid_letters) == correct_letter else 0.0

    # Option letters are never answer text: a substring test on them would credit any response
    # containing that character.
    candidate_answers = [str(metadata[key]) for key in ("correct_answer", "answer_text") if metadata.get(key)]
    if label_text and not correct_letter:
        candidate_answers.append(label_text)

    normalized_targets = {
        _normalize_text(text) for text in candidate_answers if text.strip().upper() not in valid_letters
    }
    normalized_response = f" {_normalize_text(_strip_chain_of_thought(response))} "
    for target in normalized_targets:
        if target and f" {target} " in normalized_response:
            return 1.0

    if extracted_letter and not correct_letter and label_text:
        return 1.0 if extracted_letter == label_text.strip().upper() else 0.0

    return 0.0
