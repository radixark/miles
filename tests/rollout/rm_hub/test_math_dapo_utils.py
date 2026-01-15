import pytest

from miles.rollout.rm_hub.math_dapo_utils import (
    compute_score,
    is_correct_minerva,
    is_correct_strict_box,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)


class TestLastBoxedOnlyString:
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            (r"The answer is \boxed{42}", r"\boxed{42}"),
            (r"\boxed{x^2}", r"\boxed{x^2}"),
            (r"No boxed", None),
            (r"Multiple \boxed{1} and \boxed{2}", r"\boxed{2}"),
        ],
    )
    def test_last_boxed_only_string(self, input_str, expected):
        assert last_boxed_only_string(input_str) == expected


class TestRemoveBoxed:
    def test_remove_boxed_valid(self):
        assert remove_boxed(r"\boxed{42}") == "42"
        assert remove_boxed(r"\boxed{x + 1}") == "x + 1"

    def test_remove_boxed_invalid(self):
        with pytest.raises(AssertionError):
            remove_boxed("not boxed")


class TestNormalizeFinalAnswer:
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("42", "42"),
            ("  42  ", "42"),
            (r"\text{hello}", "hello"),
            (r"\textbf{bold}", "bold"),
            (r"x = 42", "42"),
            (r"100 square", "100"),
            (r"$50$ dollars", "50"),
            (r"\boxed{42}", "42"),
            (r"\frac12", r"frac{1}{2}"),
            (r"\sqrt3", r"sqrt{3}"),
            ("1,000", "1000"),
            ("<|im_end|>", ""),
        ],
    )
    def test_normalize_final_answer(self, input_str, expected):
        assert normalize_final_answer(input_str) == expected


class TestIsCorrectMinerva:
    @pytest.mark.parametrize(
        "solution,gt,expected_correct",
        [
            ("Answer: 42", "42", True),
            ("Answer: 100", "42", False),
            ("The answer is: 5", "5", True),
            ("answer: wrong", "42", False),
        ],
    )
    def test_is_correct_minerva(self, solution, gt, expected_correct):
        correct, pred = is_correct_minerva(solution, gt)
        assert correct == expected_correct

    def test_is_correct_minerva_with_extraction(self):
        correct, pred = is_correct_minerva("Answer: 42", r"\boxed{42}", gt_need_extract=True)
        assert correct is True


class TestIsCorrectStrictBox:
    def test_correct_strict_box(self):
        score, pred = is_correct_strict_box(r"blah blah \boxed{42}", "42")
        assert score == 1
        assert pred == "42"

    def test_incorrect_strict_box(self):
        score, pred = is_correct_strict_box(r"\boxed{wrong}", "42")
        assert score == -1
        assert pred == "wrong"

    def test_no_boxed(self):
        score, pred = is_correct_strict_box("no box here", "42")
        assert score == -1
        assert pred is None


class TestComputeScore:
    def test_correct_answer(self):
        result = compute_score("Answer: 42", "42")
        assert result["score"] == 1.0
        assert result["acc"] is True
        assert result["pred"] == "42"

    def test_incorrect_answer(self):
        result = compute_score("Answer: wrong", "42")
        assert result["score"] == -1.0
        assert result["acc"] is False

    def test_strict_box_mode(self):
        result = compute_score(r"\boxed{42}", "42", strict_box_verify=True)
        assert result["score"] == 1.0

    def test_long_solution_truncated(self):
        long_solution = "x" * 500 + " Answer: 42"
        result = compute_score(long_solution, "42")
        assert result["acc"] is True
