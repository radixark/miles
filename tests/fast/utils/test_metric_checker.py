from argparse import Namespace

import pytest

from miles.utils.metric_checker import MetricChecker


def _make_checker(policy: str) -> MetricChecker:
    args = Namespace(
        ci_metric_checker_key="eval/gsm8k",
        ci_metric_checker_threshold=0.4,
        ci_metric_checker_policy=policy,
    )
    return MetricChecker(args)


class TestAnyPolicy:
    def test_one_passing_eval_satisfies_the_gate(self):
        """The default policy accepts a run when any observed eval meets the threshold."""
        checker = _make_checker("any")

        checker.on_eval({"eval/gsm8k": 0.3})
        checker.on_eval({"eval/gsm8k": 0.5})

        checker.dispose()


class TestAllPolicy:
    def test_one_failing_eval_fails_the_gate(self):
        """The all policy rejects a run when any observed eval misses the threshold."""
        checker = _make_checker("all")

        checker.on_eval({"eval/gsm8k": 0.5})
        checker.on_eval({"eval/gsm8k": 0.3})

        with pytest.raises(AssertionError, match="accuracy check failed with policy all"):
            checker.dispose()

    def test_every_passing_eval_satisfies_the_gate(self):
        """The all policy accepts a run only when every observed eval meets the threshold."""
        checker = _make_checker("all")

        checker.on_eval({"eval/gsm8k": 0.4})
        checker.on_eval({"eval/gsm8k": 0.5})

        checker.dispose()


@pytest.mark.parametrize("policy", ["any", "all"])
def test_no_eval_fails_every_policy(policy: str):
    """Every policy rejects a run that never reports the configured eval metric."""
    with pytest.raises(AssertionError, match="no metrics checked"):
        _make_checker(policy).dispose()
