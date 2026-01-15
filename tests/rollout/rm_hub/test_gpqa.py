import pytest

from miles.rollout.rm_hub.gpqa import (
    _extract_letter_from_response,
    _normalize_text,
    _strip_chain_of_thought,
    compute_gpqa_reward,
)


class TestStripChainOfThought:
    def test_with_think_tag(self):
        text = "Let me think...</think>The answer is A"
        assert _strip_chain_of_thought(text) == "The answer is A"

    def test_without_think_tag(self):
        text = "The answer is A"
        assert _strip_chain_of_thought(text) == "The answer is A"

    def test_empty_string(self):
        assert _strip_chain_of_thought("") == ""

    def test_none(self):
        assert _strip_chain_of_thought(None) == ""


class TestNormalizeText:
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("Hello World", "hello world"),
            ("Test-123", "test 123"),
            ("A, B, C", "a b c"),
            ("", ""),
        ],
    )
    def test_normalize_text(self, input_str, expected):
        assert _normalize_text(input_str) == expected


class TestExtractLetterFromResponse:
    @pytest.mark.parametrize(
        "response,expected",
        [
            ("The answer is A", "A"),
            ("answer: B", "B"),
            ("I think C is correct", "C"),
            ("final answer: D", "D"),
            ("Option A is the best choice", "A"),
            ("</think>The answer is B", "B"),
            ("After analysis, my choice is C", "C"),
        ],
    )
    def test_extract_letter(self, response, expected):
        assert _extract_letter_from_response(response, "ABCD") == expected

    def test_fallback_to_last_valid_letter(self):
        assert _extract_letter_from_response("A B C D", "ABCD") == "D"

    def test_no_valid_letter(self):
        assert _extract_letter_from_response("No valid letter here", "ABCD") is None

    def test_empty_response(self):
        assert _extract_letter_from_response("", "ABCD") is None
        assert _extract_letter_from_response(None, "ABCD") is None

    def test_invalid_letter_filtered(self):
        result = _extract_letter_from_response("The answer is Z", "ABCD")
        assert result is None


class TestComputeGpqaReward:
    def test_correct_letter_label(self):
        assert compute_gpqa_reward("Answer: A", "A") == 1.0

    def test_wrong_letter_label(self):
        assert compute_gpqa_reward("Answer: A", "B") == 0.0

    def test_none_response(self):
        assert compute_gpqa_reward(None, "A") == 0.0

    def test_with_correct_letter_in_metadata(self):
        metadata = {"correct_letter": "B"}
        assert compute_gpqa_reward("Answer: B", "ignored", metadata=metadata) == 1.0
        assert compute_gpqa_reward("Answer: A", "ignored", metadata=metadata) == 0.0

    def test_with_choices_and_index_label(self):
        metadata = {"choices": ["Option 1", "Option 2", "Option 3", "Option 4"]}
        assert compute_gpqa_reward("Answer: A", 0, metadata=metadata) == 1.0
        assert compute_gpqa_reward("Answer: B", 1, metadata=metadata) == 1.0

    def test_with_valid_letters_in_metadata(self):
        metadata = {"valid_letters": ["X", "Y", "Z"]}
        assert compute_gpqa_reward("Answer: X", "X", metadata=metadata) == 1.0
        assert compute_gpqa_reward("Answer: A", "X", metadata=metadata) == 0.0

    def test_text_matching_fallback(self):
        metadata = {"choices": ["Paris", "London", "Berlin", "Rome"], "correct_letter": "A"}
        assert compute_gpqa_reward("I believe the answer is Paris", "", metadata=metadata) == 1.0

    def test_choices_as_dict(self):
        metadata = {"choices": {"A": "Paris", "B": "London"}, "correct_letter": "A"}
        assert compute_gpqa_reward("Answer: A", "", metadata=metadata) == 1.0

    def test_label_text_matching(self):
        metadata = {"choices": ["Paris", "London", "Berlin", "Rome"]}
        assert compute_gpqa_reward("The answer is Paris", "Paris", metadata=metadata) == 1.0

    def test_cot_stripped(self):
        response = "Let me think step by step...</think>The answer is A"
        assert compute_gpqa_reward(response, "A") == 1.0
