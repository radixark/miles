import logging

logger = logging.getLogger(__name__)


class MetricChecker:
    @staticmethod
    def maybe_create(args):
        if args.ci_test and (args.ci_metric_checker_key is not None):
            return MetricChecker(args)
        return None

    def __init__(self, args):
        self.args = args
        self._check_results: list[bool] = []

    def on_eval(self, metrics: dict[str, float]):
        actual_value = metrics.get(self.args.ci_metric_checker_key)
        assert actual_value is not None, f"{metrics=} {self.args.ci_metric_checker_key=}"

        check_success = actual_value >= self.args.ci_metric_checker_threshold
        logger.info(f"[MetricChecker] {check_success=} {actual_value=} {self.args.ci_metric_checker_threshold=}")

        self._check_results.append(check_success)

    def dispose(self):
        assert self._check_results, "[MetricChecker] no metrics checked"
        if (expected := self.args.ci_metric_checker_expect_num) is None:
            assert any(self._check_results), f"[MetricChecker] accuracy check failed: {self._check_results=}"
        else:
            assert len(self._check_results) == expected and all(self._check_results), (
                f"[MetricChecker] expected exactly {expected} checks and all to succeed: " f"{self._check_results=}"
            )
        logger.info("[MetricChecker] pass dispose check")
