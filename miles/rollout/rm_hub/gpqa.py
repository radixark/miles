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


def _extract_letter_from_response(response: str, valid_letters: Iterable[str]) -> str | None:
    """
    Best-effort extraction of the selected option letter from the model response.
    """
    if not response:
        return None

    text = _strip_chain_of_thought(response)
    # Keywords are case-insensitive; the option letter must be a standalone capital,
    # so "the answer is about" does not yield "A".
    patterns = [
        r"(?i:answer|option|choice)\s*(?i:is|:)?\s*\b([A-Z])\b",
        r"\b([A-Z])\s*(?i:is\s*(?:the)?\s*correct)",
        r"(?i:final\s*(?:answer|option))\s*(?i:is|:)?\s*\b([A-Z])\b",
    ]

    valid_letters = {letter.upper() for letter in valid_letters}
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter

    # Fallback: last standalone capital letter that is valid.
    candidates = re.findall(r"\b([A-Z])\b", text)
    for letter in reversed(candidates):
        letter = letter.upper()
        if letter in valid_letters:
            return letter

    return None


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
    label_is_letter = False
    if isinstance(label, str):
        label_text = label.strip()
        label_is_letter = len(label_text) == 1 and label_text.upper() in valid_letters
        if label_is_letter and not correct_letter:
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

    candidate_answers = []
    if correct_letter and choices:
        try:
            idx = valid_letters.index(correct_letter)
        except ValueError:
            idx = None
        if idx is not None and idx < len(choices):
            candidate_answers.append(str(choices[idx]))

    for key in ("correct_answer", "answer_text"):
        value = metadata.get(key)
        if value:
            candidate_answers.append(str(value))

    # A bare option letter is not answer text; matching it would let any stray "b" score.
    if label_text and not label_is_letter:
        candidate_answers.append(label_text)

    normalized_targets = {_normalize_text(text) for text in candidate_answers if text}
    # Normalized text is space-separated tokens, so padding enforces whole-word matches.
    padded_response = f" {_normalize_text(_strip_chain_of_thought(response))} "
    for target in normalized_targets:
        if target and f" {target} " in padded_response:
            return 1.0

    if extracted_letter and not correct_letter and label_text:
        return 1.0 if extracted_letter == label_text.strip().upper() else 0.0

    return 0.0
