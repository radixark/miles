import pytest

from miles.rollout.rm_hub.f1 import f1_score, normalize_answer


class TestNormalizeAnswer:
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("Hello World", "hello world"),
            ("The quick brown fox", "quick brown fox"),
            ("A cat and a dog", "cat and dog"),
            ("Hello, world!", "hello world"),
            ("  multiple   spaces  ", "multiple spaces"),
            ("An apple", "apple"),
            ("UPPERCASE", "uppercase"),
        ],
    )
    def test_normalize_answer(self, input_str, expected):
        assert normalize_answer(input_str) == expected


class TestF1Score:
    @pytest.mark.parametrize(
        "prediction,ground_truth,expected_f1,expected_prec,expected_recall",
        [
            ("hello world", "hello world", 1.0, 1.0, 1.0),
            ("hello world foo", "hello world bar", 2 / 3, 2 / 3, 2 / 3),
            ("abc", "xyz", 0, 0, 0),
            (None, "anything", 0, 0, 0),
            ("yes", "no", 0, 0, 0),
            ("no", "yes", 0, 0, 0),
            ("yes", "yes", 1.0, 1.0, 1.0),
            ("noanswer", "yes", 0, 0, 0),
            ("the answer is correct", "answer is correct", 1.0, 1.0, 1.0),
            ("hello, world!", "hello world", 1.0, 1.0, 1.0),
            ("hello", "hello world", pytest.approx(2 / 3), 1.0, 0.5),
        ],
    )
    def test_f1_score(self, prediction, ground_truth, expected_f1, expected_prec, expected_recall):
        f1, prec, recall = f1_score(prediction, ground_truth)
        assert f1 == expected_f1
        assert prec == expected_prec
        assert recall == expected_recall
