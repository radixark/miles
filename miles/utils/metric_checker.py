from typing import Dict


class MetricChecker:
    def __init__(self, args):
        self.args = args

    def on_eval(self, metrics: Dict[str, float]):
        value = metrics[self.args.ci_metric_checker_key]
        check_success = value >= self.args.ci_metric_checker_value
        TODO
