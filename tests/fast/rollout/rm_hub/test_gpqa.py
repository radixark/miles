import pytest

from miles.rollout.rm_hub.gpqa import (
    _extract_letter_from_response,
    _normalize_text,
    _strip_chain_of_thought,
    compute_gpqa_reward,
)


class TestStripChainOfThought:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Let me think...</think>The answer is A", "The answer is A"),
            ("The answer is A", "The answer is A"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_strip_chain_of_thought(self, text, expected):
        assert _strip_chain_of_thought(text) == expected


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
            ("A B C D", "D"),
            ("No valid letter here", None),
            ("", None),
            (None, None),
            ("The answer is Z", None),
        ],
    )
    def test_extract_letter(self, response, expected):
        assert _extract_letter_from_response(response, "ABCD") == expected


class TestComputeGpqaReward:
    @pytest.mark.parametrize(
        "response,label,metadata,expected",
        [
            ("Answer: A", "A", None, 1.0),
            ("Answer: A", "B", None, 0.0),
            (None, "A", None, 0.0),
            ("Answer: B", "ignored", {"correct_letter": "B"}, 1.0),
            ("Answer: A", "ignored", {"correct_letter": "B"}, 0.0),
            ("Answer: A", 0, {"choices": ["Option 1", "Option 2", "Option 3", "Option 4"]}, 1.0),
            ("Answer: B", 1, {"choices": ["Option 1", "Option 2", "Option 3", "Option 4"]}, 1.0),
            ("Answer: X", "X", {"valid_letters": ["X", "Y", "Z"]}, 1.0),
            ("Answer: A", "X", {"valid_letters": ["X", "Y", "Z"]}, 0.0),
            (
                "I believe the answer is Paris",
                "",
                {"choices": ["Paris", "London", "Berlin", "Rome"], "correct_letter": "A"},
                1.0,
            ),
            ("Answer: A", "", {"choices": {"A": "Paris", "B": "London"}, "correct_letter": "A"}, 1.0),
            ("The answer is Paris", "Paris", {"choices": ["Paris", "London", "Berlin", "Rome"]}, 1.0),
            ("Let me think step by step...</think>The answer is A", "A", None, 1.0),
        ],
    )
    def test_compute_gpqa_reward(self, response, label, metadata, expected):
        assert compute_gpqa_reward(response, label, metadata=metadata) == expected
