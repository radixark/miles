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
    def test_exact_match(self):
        f1, prec, recall = f1_score("hello world", "hello world")
        assert f1 == 1.0
        assert prec == 1.0
        assert recall == 1.0

    def test_partial_match(self):
        f1, prec, recall = f1_score("hello world foo", "hello world bar")
        assert 0 < f1 < 1
        assert prec == 2 / 3
        assert recall == 2 / 3

    def test_no_match(self):
        assert f1_score("abc", "xyz") == (0, 0, 0)

    def test_none_prediction(self):
        assert f1_score(None, "anything") == (0, 0, 0)

    def test_yes_no_special_handling(self):
        assert f1_score("yes", "no") == (0, 0, 0)
        assert f1_score("no", "yes") == (0, 0, 0)
        assert f1_score("yes", "yes") == (1.0, 1.0, 1.0)
        assert f1_score("noanswer", "yes") == (0, 0, 0)

    def test_with_articles(self):
        f1, _, _ = f1_score("the answer is correct", "answer is correct")
        assert f1 == 1.0

    def test_with_punctuation(self):
        f1, _, _ = f1_score("hello, world!", "hello world")
        assert f1 == 1.0

    def test_subset_match(self):
        f1, prec, recall = f1_score("hello", "hello world")
        assert prec == 1.0
        assert recall == 0.5
        assert f1 == pytest.approx(2 / 3)
