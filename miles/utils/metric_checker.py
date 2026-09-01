import logging

from miles.utils.misc import should_run_periodic_action

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
        self._expected_num_checks = self._get_expected_num_checks() if args.ci_metric_checker_policy == "all" else None

    def on_eval(self, metrics: dict[str, float]):
        actual_value = metrics.get(self.args.ci_metric_checker_key)
        assert actual_value is not None, f"{metrics=} {self.args.ci_metric_checker_key=}"

        check_success = actual_value >= self.args.ci_metric_checker_threshold
        logger.info(f"[MetricChecker] {check_success=} {actual_value=} {self.args.ci_metric_checker_threshold=}")

        self._check_results.append(check_success)

    def dispose(self):
        assert self._check_results, "[MetricChecker] no metrics checked"
        match self.args.ci_metric_checker_policy:
            case "any":
                policy_succeeded = any(self._check_results)
            case "all":
                policy_succeeded = len(self._check_results) == self._expected_num_checks and all(self._check_results)
            case policy:
                raise ValueError(f"Unknown CI metric checker policy: {policy}")
        assert policy_succeeded, (
            f"[MetricChecker] accuracy check failed with policy {self.args.ci_metric_checker_policy}: "
            f"{self._check_results=}, {self._expected_num_checks=}"
        )
        logger.info("[MetricChecker] pass dispose check")

    def _get_expected_num_checks(self):
        assert self.args.num_rollout is not None, "[MetricChecker] policy all requires --num-rollout"
        start_rollout_id = self.args.start_rollout_id or 0
        baseline_count = int(start_rollout_id == 0 and not self.args.skip_eval_before_train)
        periodic_count = sum(
            should_run_periodic_action(
                rollout_id,
                self.args.eval_interval,
                num_rollout=self.args.num_rollout,
            )
            for rollout_id in range(start_rollout_id, self.args.num_rollout)
        )
        return baseline_count + periodic_count
