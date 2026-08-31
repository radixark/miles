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
        self._num_checks = 0
        self._check_success = args.ci_metric_checker_policy == "all"

    def on_eval(self, metrics: dict[str, float]) -> None:
        actual_value = metrics.get(self.args.ci_metric_checker_key)
        assert actual_value is not None, f"{metrics=} {self.args.ci_metric_checker_key=}"

        check_success = actual_value >= self.args.ci_metric_checker_threshold
        logger.info(f"[MetricChecker] {check_success=} {actual_value=} {self.args.ci_metric_checker_threshold=}")

        self._num_checks += 1
        if self.args.ci_metric_checker_policy == "any":
            self._check_success |= check_success
        else:
            self._check_success &= check_success

    def dispose(self) -> None:
        assert self._num_checks > 0, "[MetricChecker] no metrics checked"
        assert (
            self._check_success
        ), f"[MetricChecker] accuracy check failed with policy {self.args.ci_metric_checker_policy}"
        logger.info("[MetricChecker] pass dispose check")
