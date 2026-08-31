import logging
from argparse import Namespace

logger = logging.getLogger(__name__)


class MetricChecker:
    @staticmethod
    def maybe_create(args: Namespace) -> "MetricChecker | None":
        if args.ci_test and (args.ci_metric_checker_key is not None):
            return MetricChecker(args)
        return None

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self._check_success: list[bool] = []

    def on_eval(self, metrics: dict[str, float]) -> None:
        actual_value = metrics.get(self.args.ci_metric_checker_key)
        assert actual_value is not None, f"{metrics=} {self.args.ci_metric_checker_key=}"

        check_success = actual_value >= self.args.ci_metric_checker_threshold
        logger.info(f"[MetricChecker] {check_success=} {actual_value=} {self.args.ci_metric_checker_threshold=}")

        self._check_success.append(check_success)

    def dispose(self) -> None:
        assert self._check_success, "[MetricChecker] no metrics checked"
        policy_succeeded = (
            any(self._check_success) if self.args.ci_metric_checker_policy == "any" else all(self._check_success)
        )
        assert policy_succeeded, (
            f"[MetricChecker] accuracy check failed with policy {self.args.ci_metric_checker_policy}: "
            f"{self._check_success=}"
        )
        logger.info("[MetricChecker] pass dispose check")
