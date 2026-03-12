"""Tests for miles.utils.ft.controller.types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.detectors.base import _filter_node_ids_by_active
from miles.utils.ft.controller.types import (
    ActionType,
    ControllerMode,
    ControllerStatus,
    Decision,
    NodeFault,
    TriggerType,
)


class TestNodeFault:
    def test_construction(self) -> None:
        fault = NodeFault(node_id="node-1", reason="gpu error")

        assert fault.node_id == "node-1"
        assert fault.reason == "gpu error"

    def test_ephemeral_defaults_to_false(self) -> None:
        fault = NodeFault(node_id="node-0", reason="test")

        assert fault.ephemeral is False

    def test_ephemeral_can_be_set_true(self) -> None:
        fault = NodeFault(node_id="node-0", reason="test", ephemeral=True)

        assert fault.ephemeral is True

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NodeFault(node_id="node-0", reason="test", extra_field="bad")  # type: ignore[call-arg]


class TestDecision:
    def test_no_fault_construction(self) -> None:
        decision = Decision.no_fault(reason="all clear")

        assert decision.action == ActionType.NONE
        assert decision.reason == "all clear"
        assert decision.trigger is None
        assert decision.bad_node_ids == []

    def test_action_not_none_requires_trigger(self) -> None:
        with pytest.raises(ValidationError, match="trigger is required"):
            Decision(
                action=ActionType.ENTER_RECOVERY,
                reason="fault found",
                trigger=None,
            )

    def test_action_none_allows_no_trigger(self) -> None:
        decision = Decision(action=ActionType.NONE, reason="ok")

        assert decision.trigger is None

    def test_enter_recovery_with_trigger(self) -> None:
        decision = Decision(
            action=ActionType.ENTER_RECOVERY,
            reason="hang detected",
            trigger=TriggerType.HANG,
            bad_node_ids=["node-0"],
        )

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == TriggerType.HANG

    def test_notify_human_with_trigger(self) -> None:
        decision = Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="too many faults",
            trigger=TriggerType.MISC,
        )

        assert decision.action == ActionType.NOTIFY_HUMAN


class TestDecisionFromNodeFaults:
    def test_empty_faults_returns_no_fault(self) -> None:
        decision = Decision.from_node_faults(
            faults=[],
            fallback_reason="nothing found",
            trigger=TriggerType.HARDWARE,
        )

        assert decision.action == ActionType.NONE
        assert decision.reason == "nothing found"

    def test_non_ephemeral_faults_trigger_recovery(self) -> None:
        faults = [
            NodeFault(node_id="node-0", reason="gpu xid 48"),
            NodeFault(node_id="node-1", reason="gpu xid 74"),
        ]
        decision = Decision.from_node_faults(
            faults=faults,
            fallback_reason="",
            trigger=TriggerType.HARDWARE,
        )

        assert decision.action == ActionType.ENTER_RECOVERY
        assert sorted(decision.bad_node_ids) == ["node-0", "node-1"]
        assert decision.trigger == TriggerType.HARDWARE

    def test_all_ephemeral_faults_returns_no_fault(self) -> None:
        faults = [
            NodeFault(node_id="node-0", reason="transient nic flap", ephemeral=True),
            NodeFault(node_id="node-1", reason="transient nic flap", ephemeral=True),
        ]
        decision = Decision.from_node_faults(
            faults=faults,
            fallback_reason="nic checks",
            trigger=TriggerType.NETWORK,
        )

        assert decision.action == ActionType.NONE
        assert "ephemeral" in decision.reason

    def test_mixed_ephemeral_only_non_ephemeral_trigger_recovery(self) -> None:
        faults = [
            NodeFault(node_id="node-0", reason="gpu error", ephemeral=False),
            NodeFault(node_id="node-1", reason="nic flap", ephemeral=True),
        ]
        decision = Decision.from_node_faults(
            faults=faults,
            fallback_reason="",
            trigger=TriggerType.HARDWARE,
        )

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.bad_node_ids == ["node-0"]

    def test_mixed_reason_excludes_ephemeral_fault_details(self) -> None:
        """M-5: previously the reason string included ephemeral fault
        reasons alongside non-ephemeral ones, making it ambiguous which
        faults actually triggered recovery. Now only non-ephemeral
        fault reasons appear, with a count note for ephemeral ones."""
        faults = [
            NodeFault(node_id="node-0", reason="gpu xid 48", ephemeral=False),
            NodeFault(node_id="node-1", reason="transient nic flap", ephemeral=True),
        ]
        decision = Decision.from_node_faults(
            faults=faults,
            fallback_reason="",
            trigger=TriggerType.HARDWARE,
        )

        assert "gpu xid 48" in decision.reason
        assert "transient nic flap" not in decision.reason
        assert "ephemeral" in decision.reason


class TestDecisionUniqueNodeIds:
    def test_deduplicates_preserving_order(self) -> None:
        faults = [
            NodeFault(node_id="node-b", reason="r1"),
            NodeFault(node_id="node-a", reason="r2"),
            NodeFault(node_id="node-b", reason="r3"),
            NodeFault(node_id="node-c", reason="r4"),
            NodeFault(node_id="node-a", reason="r5"),
        ]

        result = Decision._unique_node_ids(faults)

        assert result == ["node-b", "node-a", "node-c"]

    def test_empty_input(self) -> None:
        assert Decision._unique_node_ids([]) == []

    def test_single_fault(self) -> None:
        faults = [NodeFault(node_id="node-0", reason="r")]
        assert Decision._unique_node_ids(faults) == ["node-0"]


class TestControllerStatus:
    def test_construction_monitoring(self) -> None:
        status = ControllerStatus(
            tick_count=5,
            active_run_id="run-1",
            latest_iteration=100,
            subsystem_states={"training": "DetectingAnomalySt"},
            recovery=None,
        )

        assert status.mode == ControllerMode.MONITORING
        assert status.tick_count == 5
        assert status.recovery_in_progress is False

    def test_construction_recovery(self) -> None:
        from miles.utils.ft.controller.types import RecoveryInfo

        recovery = RecoveryInfo(phase="reattempting", bad_nodes=["node-0"], bad_nodes_confirmed=False)
        status = ControllerStatus(
            tick_count=10,
            active_run_id="run-2",
            latest_iteration=50,
            subsystem_states={"training": "RecoveringSt"},
            recovery=recovery,
        )

        assert status.mode == ControllerMode.RECOVERY
        assert status.recovery_in_progress is True
        assert status.recovery.bad_nodes == ["node-0"]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ControllerStatus(
                tick_count=0,
                active_run_id=None,
                latest_iteration=None,
                subsystem_states={},
                recovery=None,
                bogus_field="x",  # type: ignore[call-arg]
            )


class TestFilterNodeIdsByActive:
    def test_filters_to_active_set(self) -> None:
        result = _filter_node_ids_by_active(
            node_ids=["node-0", "node-1", "node-2"],
            active_node_ids={"node-0", "node-2"},
        )

        assert result == ["node-0", "node-2"]

    def test_empty_active_returns_empty(self) -> None:
        result = _filter_node_ids_by_active(
            node_ids=["node-0", "node-1"],
            active_node_ids=set(),
        )

        assert result == []

    def test_preserves_order(self) -> None:
        result = _filter_node_ids_by_active(
            node_ids=["node-c", "node-a", "node-b"],
            active_node_ids={"node-a", "node-b", "node-c"},
        )

        assert result == ["node-c", "node-a", "node-b"]
