from types import SimpleNamespace

import pytest

from miles.utils.metric_checker import MetricChecker


def test_on_eval_reports_cumulative_success() -> None:
    checker = MetricChecker(
        SimpleNamespace(
            ci_metric_checker_key="eval/gsm8k",
            ci_metric_checker_threshold=0.55,
        )
    )

    assert checker.on_eval({"eval/gsm8k": 0.54}) is False
    assert checker.on_eval({"eval/gsm8k": 0.56}) is True
    assert checker.on_eval({"eval/gsm8k": 0.53}) is True
    checker.dispose()


def test_dispose_rejects_runs_without_a_successful_eval() -> None:
    checker = MetricChecker(
        SimpleNamespace(
            ci_metric_checker_key="eval/gsm8k",
            ci_metric_checker_threshold=0.55,
        )
    )

    checker.on_eval({"eval/gsm8k": 0.54})

    with pytest.raises(AssertionError, match=r"accuracy check failed"):
        checker.dispose()
