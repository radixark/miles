from typing import Dict


class MetricChecker:
    def __init__(self, args):
        self.args = args
        self._exists_check_success = False

    def on_eval(self, metrics: Dict[str, float]):
        actual_value = metrics.get(self.args.ci_metric_checker_key)
        assert actual_value is not None, f"{metrics=} {self.args.ci_metric_checker_key=}"

        check_success = actual_value >= self.args.ci_metric_checker_value
        self._exists_check_success |= check_success
        print(f"[MetricChecker] {check_success=} {actual_value=} {self.args.ci_metric_checker_value}")

    def dispose(self):
        assert self._exists_check_success, "[MetricChecker] accuracy check failed"
