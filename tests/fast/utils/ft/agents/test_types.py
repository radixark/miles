import pytest
from pydantic import ValidationError

from miles.utils.ft.agents.types import DiagnosticResult
from miles.utils.ft.controller.types import ActionType, Decision, NodeFault, TriggerType
from miles.utils.ft.agents.types import CollectorOutput, CounterSample, GaugeSample
from miles.utils.ft.adapters.types import ft_controller_actor_name


class TestGaugeSample:
    def test_normal_construction(self) -> None:
        sample = GaugeSample(
            name="gpu_temperature_celsius",
            labels={"gpu": "0"},
            value=75.0,
        )
        assert sample.name == "gpu_temperature_celsius"
        assert sample.labels == {"gpu": "0"}
        assert sample.value == 75.0
        assert sample.metric_type == "gauge"

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            GaugeSample(name="test", labels={"a": "b"})  # type: ignore[call-arg]

    def test_empty_labels(self) -> None:
        sample = GaugeSample(name="metric", labels={}, value=42.0)
        assert sample.labels == {}

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            GaugeSample(
                name="test",
                labels={},
                value=1.0,
                unknown_field="oops",  # type: ignore[call-arg]
            )


class TestCounterSample:
    def test_normal_construction(self) -> None:
        sample = CounterSample(
            name="xid_count_total",
            labels={},
            delta=3.0,
        )
        assert sample.name == "xid_count_total"
        assert sample.labels == {}
        assert sample.delta == 3.0
        assert sample.metric_type == "counter"

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            CounterSample(name="test", labels={})  # type: ignore[call-arg]

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            CounterSample(
                name="test",
                labels={},
                delta=1.0,
                unknown_field="oops",  # type: ignore[call-arg]
            )


class TestFtBaseModelExtraForbid:
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

    def test_with_gauge_metrics(self) -> None:
        sample = GaugeSample(name="gpu_available", labels={"gpu": "0"}, value=1.0)
        output = CollectorOutput(metrics=[sample])
        assert len(output.metrics) == 1
        assert output.metrics[0].name == "gpu_available"

    def test_with_mixed_metrics(self) -> None:
        gauge = GaugeSample(name="gpu_temp", labels={"gpu": "0"}, value=72.0)
        counter = CounterSample(name="xid_count", labels={}, delta=3.0)
        output = CollectorOutput(metrics=[gauge, counter])
        assert len(output.metrics) == 2
        assert isinstance(output.metrics[0], GaugeSample)
        assert isinstance(output.metrics[1], CounterSample)


class TestDecision:
    def test_action_none_without_trigger_is_valid(self) -> None:
        decision = Decision(action=ActionType.NONE, reason="test")
        assert decision.action == ActionType.NONE
        assert decision.trigger is None

    def test_non_none_action_without_trigger_raises(self) -> None:
        with pytest.raises(ValidationError, match="trigger is required"):
            Decision(action=ActionType.ENTER_RECOVERY, reason="test")

    def test_non_none_action_with_trigger_is_valid(self) -> None:
        for action in ActionType:
            if action == ActionType.NONE:
                continue
            decision = Decision(action=action, reason="test", trigger=TriggerType.CRASH)
            assert decision.action == action
            assert decision.trigger == TriggerType.CRASH

    def test_invalid_action_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            Decision(action="invalid_action", reason="test")  # type: ignore[arg-type]

    def test_default_fields(self) -> None:
        decision = Decision(action=ActionType.NONE, reason="all clear")
        assert decision.bad_node_ids == []
        assert decision.trigger is None

    def test_mark_bad_with_nodes(self) -> None:
        decision = Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=["node-0", "node-1"],
            reason="GPU lost on multiple nodes",
            trigger=TriggerType.HARDWARE,
        )
        assert decision.bad_node_ids == ["node-0", "node-1"]


class TestDecisionFromNodeFaults:
    def test_empty_faults_returns_none_action(self) -> None:
        decision = Decision.from_node_faults(
            faults=[],
            fallback_reason="no faults",
            trigger=TriggerType.HARDWARE,
        )

        assert decision.action == ActionType.NONE
        assert decision.reason == "no faults"
        assert decision.bad_node_ids == []
        assert decision.trigger is None

    def test_single_fault(self) -> None:
        faults = [NodeFault(node_id="node-0", reason="GPU lost")]
        decision = Decision.from_node_faults(
            faults=faults,
            fallback_reason="n/a",
            trigger=TriggerType.HARDWARE,
        )

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.bad_node_ids == ["node-0"]
        assert decision.reason == "GPU lost"
        assert decision.trigger == TriggerType.HARDWARE

    def test_multiple_faults_different_nodes_sorted(self) -> None:
        faults = [
            NodeFault(node_id="node-2", reason="NIC down"),
            NodeFault(node_id="node-0", reason="GPU ECC error"),
        ]
        decision = Decision.from_node_faults(
            faults=faults,
            fallback_reason="n/a",
            trigger=TriggerType.NETWORK,
        )

        assert decision.bad_node_ids == ["node-0", "node-2"]
        assert "NIC down" in decision.reason
        assert "GPU ECC error" in decision.reason

    def test_duplicate_node_ids_deduplicated(self) -> None:
        faults = [
            NodeFault(node_id="node-0", reason="GPU 0 lost"),
            NodeFault(node_id="node-0", reason="GPU 1 lost"),
        ]
        decision = Decision.from_node_faults(
            faults=faults,
            fallback_reason="n/a",
            trigger=TriggerType.HARDWARE,
        )

        assert decision.bad_node_ids == ["node-0"]
        assert decision.reason == "GPU 0 lost; GPU 1 lost"

    def test_reason_semicolon_joined(self) -> None:
        faults = [
            NodeFault(node_id="node-0", reason="reason-a"),
            NodeFault(node_id="node-1", reason="reason-b"),
            NodeFault(node_id="node-2", reason="reason-c"),
        ]
        decision = Decision.from_node_faults(
            faults=faults,
            fallback_reason="n/a",
            trigger=TriggerType.CRASH,
        )

        assert decision.reason == "reason-a; reason-b; reason-c"


class TestDecisionFromNodeFaultsEphemeral:
    def test_excludes_ephemeral_from_bad_node_ids(self) -> None:
        """Mixed ephemeral + non-ephemeral: only non-ephemeral in bad_node_ids."""
        faults = [
            NodeFault(node_id="node-0", reason="gpu error", ephemeral=False),
            NodeFault(node_id="node-1", reason="nic flap", ephemeral=True),
        ]
        decision = Decision.from_node_faults(
            faults=faults,
            fallback_reason="n/a",
            trigger=TriggerType.HARDWARE,
        )
        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.bad_node_ids == ["node-0"]
        assert "gpu error" in decision.reason
        assert "nic flap" in decision.reason

    def test_ephemeral_only_returns_none_action(self) -> None:
        """All faults ephemeral -> NONE action (completely ignored)."""
        faults = [
            NodeFault(node_id="node-0", reason="nic flap 1", ephemeral=True),
            NodeFault(node_id="node-1", reason="nic flap 2", ephemeral=True),
        ]
        decision = Decision.from_node_faults(
            faults=faults,
            fallback_reason="NIC alerts below threshold",
            trigger=TriggerType.NETWORK,
        )
        assert decision.action == ActionType.NONE
        assert "ephemeral only" in decision.reason
        assert decision.bad_node_ids == []

    def test_mixed_ephemeral_keeps_non_ephemeral(self) -> None:
        """Two non-ephemeral + one ephemeral -> only non-ephemeral in bad_node_ids."""
        faults = [
            NodeFault(node_id="node-A", reason="disk error", ephemeral=False),
            NodeFault(node_id="node-B", reason="transient glitch", ephemeral=True),
            NodeFault(node_id="node-C", reason="gpu down", ephemeral=False),
        ]
        decision = Decision.from_node_faults(
            faults=faults,
            fallback_reason="n/a",
            trigger=TriggerType.HARDWARE,
        )
        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.bad_node_ids == ["node-A", "node-C"]


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
            diagnostic_type="nccl_simple",
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
