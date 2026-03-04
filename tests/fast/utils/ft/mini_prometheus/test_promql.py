from datetime import timedelta

import pytest

from miles.utils.ft.controller.mini_prometheus.promql import (
    CompareExpr,
    CompareOp,
    LabelMatchOp,
    LabelMatcher,
    MetricSelector,
    RangeFunction,
    RangeFunctionCompare,
    match_labels,
    parse_promql,
)


class TestParsePromQL:
    def test_simple_metric_name(self) -> None:
        expr = parse_promql("miles_ft_node_gpu_available")
        assert isinstance(expr, MetricSelector)
        assert expr.name == "miles_ft_node_gpu_available"
        assert expr.matchers == []

    def test_metric_with_label_filter(self) -> None:
        expr = parse_promql('miles_ft_node_xid_code_recent{xid="48"}')
        assert isinstance(expr, MetricSelector)
        assert expr.name == "miles_ft_node_xid_code_recent"
        assert len(expr.matchers) == 1
        assert expr.matchers[0].label == "xid"
        assert expr.matchers[0].op == LabelMatchOp.EQ
        assert expr.matchers[0].value == "48"

    def test_metric_with_neq_label(self) -> None:
        expr = parse_promql('gpu_available{node_id!="node-0"}')
        assert isinstance(expr, MetricSelector)
        assert expr.matchers[0].op == LabelMatchOp.NEQ

    def test_metric_with_regex_label(self) -> None:
        expr = parse_promql('gpu_available{node_id=~"node-.*"}')
        assert isinstance(expr, MetricSelector)
        assert expr.matchers[0].op == LabelMatchOp.RE
        assert expr.matchers[0].value == "node-.*"

    def test_compare_eq(self) -> None:
        expr = parse_promql("miles_ft_node_gpu_available == 0")
        assert isinstance(expr, CompareExpr)
        assert expr.selector.name == "miles_ft_node_gpu_available"
        assert expr.op == CompareOp.EQ
        assert expr.threshold == 0.0

    def test_compare_gt(self) -> None:
        expr = parse_promql("gpu_temperature_celsius > 90")
        assert isinstance(expr, CompareExpr)
        assert expr.op == CompareOp.GT
        assert expr.threshold == 90.0

    def test_compare_lte(self) -> None:
        expr = parse_promql("disk_available_bytes <= 1000000")
        assert isinstance(expr, CompareExpr)
        assert expr.op == CompareOp.LTE

    def test_range_function_count_over_time(self) -> None:
        expr = parse_promql("count_over_time(nic_alert[5m])")
        assert isinstance(expr, RangeFunction)
        assert expr.func_name == "count_over_time"
        assert expr.selector.name == "nic_alert"
        assert expr.duration == timedelta(minutes=5)

    def test_range_function_changes(self) -> None:
        expr = parse_promql("changes(training_iteration[10m])")
        assert isinstance(expr, RangeFunction)
        assert expr.func_name == "changes"
        assert expr.duration == timedelta(minutes=10)

    def test_range_function_with_compare(self) -> None:
        expr = parse_promql("count_over_time(nic_alert[5m]) >= 2")
        assert isinstance(expr, RangeFunctionCompare)
        assert expr.func.func_name == "count_over_time"
        assert expr.op == CompareOp.GTE
        assert expr.threshold == 2.0

    def test_changes_with_compare(self) -> None:
        expr = parse_promql("changes(training_iteration[10m]) == 0")
        assert isinstance(expr, RangeFunctionCompare)
        assert expr.func.func_name == "changes"
        assert expr.op == CompareOp.EQ
        assert expr.threshold == 0.0

    def test_range_function_with_labels(self) -> None:
        expr = parse_promql('count_over_time(xid_code_recent{xid="48"}[5m])')
        assert isinstance(expr, RangeFunction)
        assert expr.selector.name == "xid_code_recent"
        assert len(expr.selector.matchers) == 1
        assert expr.selector.matchers[0].value == "48"

    def test_duration_seconds(self) -> None:
        expr = parse_promql("count_over_time(metric[30s])")
        assert isinstance(expr, RangeFunction)
        assert expr.duration == timedelta(seconds=30)

    def test_duration_hours(self) -> None:
        expr = parse_promql("count_over_time(metric[1h])")
        assert isinstance(expr, RangeFunction)
        assert expr.duration == timedelta(hours=1)

    def test_duration_days(self) -> None:
        expr = parse_promql("count_over_time(metric[1d])")
        assert isinstance(expr, RangeFunction)
        assert expr.duration == timedelta(days=1)

    def test_compare_with_label_selector(self) -> None:
        expr = parse_promql('gpu_temp{gpu="0"} > 90')
        assert isinstance(expr, CompareExpr)
        assert expr.selector.name == "gpu_temp"
        assert len(expr.selector.matchers) == 1
        assert expr.selector.matchers[0].label == "gpu"
        assert expr.selector.matchers[0].value == "0"
        assert expr.op == CompareOp.GT
        assert expr.threshold == 90.0

    def test_compare_neq_with_label_selector(self) -> None:
        expr = parse_promql('gpu_available{node_id!="node-0"} == 0')
        assert isinstance(expr, CompareExpr)
        assert expr.selector.name == "gpu_available"
        assert expr.selector.matchers[0].op == LabelMatchOp.NEQ
        assert expr.op == CompareOp.EQ
        assert expr.threshold == 0.0

    def test_multiple_label_matchers(self) -> None:
        expr = parse_promql('metric{env="prod", region="us"}')
        assert isinstance(expr, MetricSelector)
        assert len(expr.matchers) == 2
        assert expr.matchers[0].label == "env"
        assert expr.matchers[0].value == "prod"
        assert expr.matchers[1].label == "region"
        assert expr.matchers[1].value == "us"


class TestParsePromQLErrors:
    def test_unmatched_paren_raises(self) -> None:
        with pytest.raises(ValueError, match="Unmatched parenthesis"):
            parse_promql("count_over_time(nic_alert[5m]")

    def test_missing_range_selector_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing range selector"):
            parse_promql("count_over_time(nic_alert)")

    def test_invalid_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_promql("count_over_time(metric[5x])")


class TestMatchLabels:
    def test_eq_match(self) -> None:
        labels = {"gpu": "0", "node_id": "node-0"}
        matchers = [LabelMatcher(label="gpu", op=LabelMatchOp.EQ, value="0")]
        assert match_labels(labels, matchers) is True

    def test_eq_mismatch(self) -> None:
        labels = {"gpu": "1"}
        matchers = [LabelMatcher(label="gpu", op=LabelMatchOp.EQ, value="0")]
        assert match_labels(labels, matchers) is False

    def test_neq(self) -> None:
        labels = {"gpu": "1"}
        matchers = [LabelMatcher(label="gpu", op=LabelMatchOp.NEQ, value="0")]
        assert match_labels(labels, matchers) is True

    def test_neq_blocks_match(self) -> None:
        labels = {"gpu": "0"}
        matchers = [LabelMatcher(label="gpu", op=LabelMatchOp.NEQ, value="0")]
        assert match_labels(labels, matchers) is False

    def test_regex_match(self) -> None:
        labels = {"node_id": "node-42"}
        matchers = [LabelMatcher(label="node_id", op=LabelMatchOp.RE, value="node-.*")]
        assert match_labels(labels, matchers) is True

    def test_regex_no_match(self) -> None:
        labels = {"node_id": "worker-1"}
        matchers = [LabelMatcher(label="node_id", op=LabelMatchOp.RE, value="node-.*")]
        assert match_labels(labels, matchers) is False

    def test_missing_label_returns_empty_string(self) -> None:
        labels = {"gpu": "0"}
        matchers = [LabelMatcher(label="missing_key", op=LabelMatchOp.EQ, value="")]
        assert match_labels(labels, matchers) is True

    def test_missing_label_neq(self) -> None:
        labels = {"gpu": "0"}
        matchers = [LabelMatcher(label="missing_key", op=LabelMatchOp.EQ, value="something")]
        assert match_labels(labels, matchers) is False

    def test_multiple_matchers_all_must_pass(self) -> None:
        labels = {"gpu": "0", "node_id": "node-1"}
        matchers = [
            LabelMatcher(label="gpu", op=LabelMatchOp.EQ, value="0"),
            LabelMatcher(label="node_id", op=LabelMatchOp.EQ, value="node-1"),
        ]
        assert match_labels(labels, matchers) is True

    def test_multiple_matchers_one_fails(self) -> None:
        labels = {"gpu": "0", "node_id": "node-1"}
        matchers = [
            LabelMatcher(label="gpu", op=LabelMatchOp.EQ, value="0"),
            LabelMatcher(label="node_id", op=LabelMatchOp.EQ, value="node-2"),
        ]
        assert match_labels(labels, matchers) is False

    def test_empty_matchers_matches_all(self) -> None:
        labels = {"gpu": "0"}
        assert match_labels(labels, []) is True
