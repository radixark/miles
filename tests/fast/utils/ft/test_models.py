import pytest
from pydantic import ValidationError

from miles.utils.ft.models import (
    ActionType,
    CollectorOutput,
    Decision,
    DiagnosticResult,
    FtBaseModel,
    MetricSample,
)


class TestMetricSample:
    def test_normal_construction(self) -> None:
        sample = MetricSample(
            name="gpu_temperature_celsius",
            labels={"gpu": "0"},
            value=75.0,
        )
        assert sample.name == "gpu_temperature_celsius"
        assert sample.labels == {"gpu": "0"}
        assert sample.value == 75.0

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetricSample(name="test", labels={"a": "b"})  # type: ignore[call-arg]

    def test_empty_labels(self) -> None:
        sample = MetricSample(name="xid_count_recent", labels={}, value=3.0)
        assert sample.labels == {}


class TestFtBaseModelExtraForbid:
    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetricSample(
                name="test",
                labels={},
                value=1.0,
                unknown_field="oops",  # type: ignore[call-arg]
            )

    def test_extra_forbid_on_subclass(self) -> None:
        with pytest.raises(ValidationError):
            Decision(
                action=ActionType.NONE,
                reason="ok",
                extra="bad",  # type: ignore[call-arg]
            )


class TestCollectorOutput:
    def test_empty_metrics_is_valid(self) -> None:
        output = CollectorOutput(metrics=[])
        assert output.metrics == []

    def test_with_metrics(self) -> None:
        sample = MetricSample(name="gpu_available", labels={"gpu": "0"}, value=1.0)
        output = CollectorOutput(metrics=[sample])
        assert len(output.metrics) == 1
        assert output.metrics[0].name == "gpu_available"


class TestDecision:
    def test_action_type_enum_values(self) -> None:
        for action in ActionType:
            decision = Decision(action=action, reason="test")
            assert decision.action == action

    def test_invalid_action_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            Decision(action="invalid_action", reason="test")  # type: ignore[arg-type]

    def test_default_fields(self) -> None:
        decision = Decision(action=ActionType.NONE, reason="all clear")
        assert decision.bad_node_ids == []
        assert decision.trigger == ""

    def test_mark_bad_with_nodes(self) -> None:
        decision = Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=["node-0", "node-1"],
            reason="GPU lost on multiple nodes",
        )
        assert decision.bad_node_ids == ["node-0", "node-1"]


class TestDiagnosticResult:
    def test_normal_construction(self) -> None:
        result = DiagnosticResult(
            diagnostic_type="gpu_eud",
            node_id="node-0",
            passed=True,
            details="All GPUs passed EUD test",
        )
        assert result.diagnostic_type == "gpu_eud"
        assert result.node_id == "node-0"
        assert result.passed is True

    def test_failed_diagnostic(self) -> None:
        result = DiagnosticResult(
            diagnostic_type="intra_machine",
            node_id="node-1",
            passed=False,
            details="GPU 3 bandwidth below threshold: 120 GB/s vs 180 GB/s baseline",
        )
        assert result.passed is False
