from argparse import Namespace

import pytest

from miles.utils.metric_checker import MetricChecker


def _make_checker(expect_num: int | None = None, *, early_stop: bool = False) -> MetricChecker:
    args = Namespace(
        ci_metric_checker_key="eval/gsm8k",
        ci_metric_checker_threshold=0.4,
        ci_metric_checker_expect_num=expect_num,
        ci_metric_checker_early_stop=early_stop,
    )
    return MetricChecker(args)


class TestDefaultExpectation:
    def test_one_passing_eval_satisfies_the_gate(self):
        """The default expectation accepts a run when any observed eval meets the threshold."""
        checker = _make_checker()

        checker.on_eval({"eval/gsm8k": 0.3})
        checker.on_eval({"eval/gsm8k": 0.5})

        checker.dispose()


class TestExactExpectation:
    def test_one_failing_eval_fails_the_gate(self):
        """An exact expectation rejects the run when any observed eval misses the threshold."""
        checker = _make_checker(expect_num=2)

        checker.on_eval({"eval/gsm8k": 0.5})
        checker.on_eval({"eval/gsm8k": 0.3})

        with pytest.raises(AssertionError, match="expected exactly 2 checks and all to succeed"):
            checker.dispose()

    def test_every_passing_eval_satisfies_the_gate(self):
        """An exact expectation passes when it receives that many successful checks."""
        checker = _make_checker(expect_num=2)

        checker.on_eval({"eval/gsm8k": 0.4})
        checker.on_eval({"eval/gsm8k": 0.5})

        checker.dispose()

    @pytest.mark.parametrize("num_results", [1, 3])
    def test_a_non_exact_result_count_fails_the_gate(self, num_results: int):
        """An exact expectation rejects both missing and additional successful checks."""
        checker = _make_checker(expect_num=2)

        for _ in range(num_results):
            checker.on_eval({"eval/gsm8k": 0.5})

        with pytest.raises(AssertionError, match="expected exactly 2 checks and all to succeed"):
            checker.dispose()


@pytest.mark.parametrize("expect_num", [None, 2])
def test_no_eval_always_fails(expect_num: int | None):
    """The checker rejects a run with no eval under either expectation mode."""
    with pytest.raises(AssertionError, match="no metrics checked"):
        _make_checker(expect_num=expect_num).dispose()


class TestEarlyStop:
    def test_threshold_crossing_requests_stop_and_remains_satisfied(self):
        checker = _make_checker(early_stop=True)
        assert checker.on_eval({"eval/gsm8k": 0.39}) is False
        assert checker.on_eval({"eval/gsm8k": 0.4}) is True
        assert checker.on_eval({"eval/gsm8k": 0.3}) is True
        checker.dispose()

    def test_disabled_early_stop_preserves_the_full_run(self):
        checker = _make_checker()
        assert checker.on_eval({"eval/gsm8k": 0.5}) is False
        checker.dispose()

    def test_never_reaching_threshold_still_fails(self):
        checker = _make_checker(early_stop=True)
        assert checker.on_eval({"eval/gsm8k": 0.3}) is False
        with pytest.raises(AssertionError, match="accuracy check failed"):
            checker.dispose()
