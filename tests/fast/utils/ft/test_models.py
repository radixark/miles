import pytest
from pydantic import ValidationError

from miles.utils.ft.models import (
    ActionType,
    CollectorOutput,
    Decision,
    DiagnosticResult,
    FtBaseModel,
    MetricSample,
    NodeFault,
    TriggerType,
    ft_controller_actor_name,
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
        sample = MetricSample(name="xid_count_total", labels={}, value=3.0, metric_type="counter")
        assert sample.labels == {}
        assert sample.metric_type == "counter"


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
        assert decision.trigger == TriggerType.NONE

    def test_mark_bad_with_nodes(self) -> None:
        decision = Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=["node-0", "node-1"],
            reason="GPU lost on multiple nodes",
        )
        assert decision.bad_node_ids == ["node-0", "node-1"]


class TestDecisionFromNodeFaults:
    def test_empty_faults_returns_none_action(self) -> None:
        decision = Decision.from_node_faults(faults=[], fallback_reason="no faults")

        assert decision.action == ActionType.NONE
        assert decision.reason == "no faults"
        assert decision.bad_node_ids == []

    def test_single_fault(self) -> None:
        faults = [NodeFault(node_id="node-0", reason="GPU lost")]
        decision = Decision.from_node_faults(faults=faults, fallback_reason="n/a")

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert decision.bad_node_ids == ["node-0"]
        assert decision.reason == "GPU lost"

    def test_multiple_faults_different_nodes_sorted(self) -> None:
        faults = [
            NodeFault(node_id="node-2", reason="NIC down"),
            NodeFault(node_id="node-0", reason="GPU ECC error"),
        ]
        decision = Decision.from_node_faults(faults=faults, fallback_reason="n/a")

        assert decision.bad_node_ids == ["node-0", "node-2"]
        assert "NIC down" in decision.reason
        assert "GPU ECC error" in decision.reason

    def test_duplicate_node_ids_deduplicated(self) -> None:
        faults = [
            NodeFault(node_id="node-0", reason="GPU 0 lost"),
            NodeFault(node_id="node-0", reason="GPU 1 lost"),
        ]
        decision = Decision.from_node_faults(faults=faults, fallback_reason="n/a")

        assert decision.bad_node_ids == ["node-0"]
        assert decision.reason == "GPU 0 lost; GPU 1 lost"

    def test_reason_semicolon_joined(self) -> None:
        faults = [
            NodeFault(node_id="node-0", reason="reason-a"),
            NodeFault(node_id="node-1", reason="reason-b"),
            NodeFault(node_id="node-2", reason="reason-c"),
        ]
        decision = Decision.from_node_faults(faults=faults, fallback_reason="n/a")

        assert decision.reason == "reason-a; reason-b; reason-c"


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


class TestFtControllerActorName:
    def test_empty_ft_id_returns_default(self) -> None:
        assert ft_controller_actor_name("") == "ft_controller"

    def test_ft_id_appended_as_suffix(self) -> None:
        assert ft_controller_actor_name("abc123") == "ft_controller_abc123"

    def test_different_ids_produce_different_names(self) -> None:
        name_a = ft_controller_actor_name("exp_a")
        name_b = ft_controller_actor_name("exp_b")
        assert name_a != name_b
