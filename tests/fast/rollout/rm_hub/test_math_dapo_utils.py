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
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            (r"\boxed{42}", "42"),
            (r"\boxed{x + 1}", "x + 1"),
        ],
    )
    def test_remove_boxed_valid(self, input_str, expected):
        assert remove_boxed(input_str) == expected

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
            (r"\frac12", r"\frac{1}{2}"),
            (r"\sqrt3", r"\sqrt{3}"),
            ("1,000", "1000"),
            ("<|im_end|>", ""),
        ],
    )
    def test_normalize_final_answer(self, input_str, expected):
        assert normalize_final_answer(input_str) == expected


class TestIsCorrectMinerva:
    @pytest.mark.parametrize(
        "solution,gt,gt_need_extract,expected_correct",
        [
            ("Answer: 42", "42", False, True),
            ("Answer: 100", "42", False, False),
            ("Answer: wrong", "42", False, False),
            ("Answer: 42", r"\boxed{42}", True, True),
        ],
    )
    def test_is_correct_minerva(self, solution, gt, gt_need_extract, expected_correct):
        correct, pred = is_correct_minerva(solution, gt, gt_need_extract=gt_need_extract)
        assert correct == expected_correct


class TestIsCorrectStrictBox:
    @pytest.mark.parametrize(
        "pred,gt,expected_score,expected_pred",
        [
            (r"blah blah \boxed{42}", "42", 1, "42"),
            (r"\boxed{wrong}", "42", -1, "wrong"),
            ("no box here", "42", -1, None),
        ],
    )
    def test_is_correct_strict_box(self, pred, gt, expected_score, expected_pred):
        score, extracted = is_correct_strict_box(pred, gt)
        assert score == expected_score
        assert extracted == expected_pred


class TestComputeScore:
    @pytest.mark.parametrize(
        "solution,gt,strict_box,expected_score,expected_acc",
        [
            ("Answer: 42", "42", False, 1.0, True),
            ("Answer: wrong", "42", False, -1.0, False),
            (r"\boxed{42}", "42", True, 1.0, True),
            ("x" * 500 + " Answer: 42", "42", False, 1.0, True),
        ],
    )
    def test_compute_score(self, solution, gt, strict_box, expected_score, expected_acc):
        result = compute_score(solution, gt, strict_box_verify=strict_box)
        assert result["score"] == expected_score
        assert result["acc"] == expected_acc
