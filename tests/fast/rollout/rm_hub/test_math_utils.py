import pytest

from miles.rollout.rm_hub.math_utils import (
    _normalize,
    extract_answer,
    grade_answer_mathd,
    grade_answer_sympy,
    grade_answer_verl,
    last_boxed_only_string,
    remove_boxed,
)


class TestLastBoxedOnlyString:
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            (r"The answer is \boxed{42}", r"\boxed{42}"),
            (r"\boxed{x^2 + 1}", r"\boxed{x^2 + 1}"),
            (r"So \boxed{\frac{1}{2}}", r"\boxed{\frac{1}{2}}"),
            (r"No boxed here", None),
            (r"Multiple \boxed{1} and \boxed{2}", r"\boxed{2}"),
            (r"\boxed{nested {braces}}", r"\boxed{nested {braces}}"),
            (r"\fbox{fbox content}", r"\fbox{fbox content}"),
            ("", None),
        ],
    )
    def test_last_boxed_only_string(self, input_str, expected):
        assert last_boxed_only_string(input_str) == expected


class TestRemoveBoxed:
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            (r"\boxed{42}", "42"),
            (r"\boxed{x^2 + 1}", "x^2 + 1"),
            (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
            ("not boxed", None),
        ],
    )
    def test_remove_boxed(self, input_str, expected):
        assert remove_boxed(input_str) == expected


class TestExtractAnswer:
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            (r"The answer is \boxed{42}", "42"),
            (r"So \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
            (r"Multiple \boxed{1} then \boxed{final}", "final"),
            (r"No boxed here", None),
            ("", None),
        ],
    )
    def test_extract_answer(self, input_str, expected):
        assert extract_answer(input_str) == expected


class TestNormalize:
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("1,000", "1000"),
            (r"\text{hello}", "hello"),
            ("  42  ", "42"),
            (r"100%", "100"),
            (r"\$50", "50"),
            ("HELLO", "hello"),
            ("1,234,567", "1234567"),
            (None, None),
        ],
    )
    def test_normalize(self, input_str, expected):
        assert _normalize(input_str) == expected


class TestGradeAnswerMathd:
    @pytest.mark.parametrize(
        "given,ground_truth,expected",
        [
            ("42", "42", True),
            ("  42  ", "42", True),
            (r"\frac{1}{2}", r"\frac{1}{2}", True),
            ("wrong", "42", False),
            ("", "42", False),
        ],
    )
    def test_grade_answer_mathd(self, given, ground_truth, expected):
        assert grade_answer_mathd(given, ground_truth) == expected


class TestGradeAnswerSympy:
    @pytest.mark.parametrize(
        "given,ground_truth,expected",
        [
            ("42", "42", True),
            ("x^2", "x^2", True),
            ("1/2", "0.5", True),
            (r"\frac{1}{2}", "0.5", True),
            ("wrong", "42", False),
            ("", "42", False),
            ("(1,2)", "(1,2)", True),
            ("(1,2,3)", "(1,2)", False),
            ("42", None, False),
        ],
    )
    def test_grade_answer_sympy(self, given, ground_truth, expected):
        assert grade_answer_sympy(given, ground_truth) == expected


class TestGradeAnswerVerl:
    @pytest.mark.parametrize(
        "solution,ground_truth,expected",
        [
            (r"\boxed{42}", "42", True),
            (r"The answer is \boxed{42}", "42", True),
            (r"\boxed{1/2}", r"\frac{1}{2}", True),
            (r"\boxed{wrong}", "42", False),
            ("no boxed", "42", False),
            (r"\boxed{42}", r"\boxed{42}", True),
            ("", "42", False),
            (r"\boxed{42}", "", False),
            (r"\boxed{42}", None, False),
        ],
    )
    def test_grade_answer_verl(self, solution, ground_truth, expected):
        assert grade_answer_verl(solution, ground_truth) == expected
