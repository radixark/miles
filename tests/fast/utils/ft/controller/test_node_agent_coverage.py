from __future__ import annotations

import logging

import pytest

from miles.utils.ft.controller.node_agents import NodeAgentCoverageChecker


class TestNodeAgentCoverageChecker:
    def test_no_warning_below_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        """Uncovered nodes below threshold do not trigger a warning."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=3)

        with caplog.at_level(logging.WARNING):
            checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
            checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())

        assert "without node agent" not in caplog.text

    def test_warning_at_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning fires exactly when the uncovered count reaches the threshold."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=3)

        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())

        assert "without node agent" in caplog.text
        assert "n1" in caplog.text

    def test_no_duplicate_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Once warned, the same node does not trigger repeated warnings."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())

        warning_count = caplog.text.count("without node agent")
        assert warning_count == 1

    def test_coverage_restored_clears_alert(self, caplog: pytest.LogCaptureFixture) -> None:
        """Restoring coverage logs info and allows re-alerting if coverage drops again."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        with caplog.at_level(logging.INFO):
            checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
            checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
            checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids={"n1"})

        assert "coverage restored" in caplog.text

    def test_realert_after_restore_and_drop(self, caplog: pytest.LogCaptureFixture) -> None:
        """After coverage is restored and drops again, a new warning fires."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        with caplog.at_level(logging.WARNING):
            checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
            checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
            checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids={"n1"})
            checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
            checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())

        warning_count = caplog.text.count("without node agent")
        assert warning_count == 2

    def test_multiple_nodes_tracked_independently(self, caplog: pytest.LogCaptureFixture) -> None:
        """Each node's uncovered window is independent."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        with caplog.at_level(logging.WARNING):
            checker.check(subsystem_name="training", subsystem_node_ids={"n1", "n2"}, registered_agent_node_ids={"n2"})
            checker.check(subsystem_name="training", subsystem_node_ids={"n1", "n2"}, registered_agent_node_ids={"n2"})

        warning_lines = [r.message for r in caplog.records if "without node agent" in r.message]
        assert len(warning_lines) == 1
        assert "n1" in warning_lines[0]

    def test_oscillating_coverage_rearms_notification_each_cycle(self, caplog: pytest.LogCaptureFixture) -> None:
        """Three full oscillation cycles: each drop after restore triggers a new warning."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
                checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
                checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids={"n1"})

        warning_count = caplog.text.count("without node agent")
        assert warning_count == 3

    def test_node_not_in_training_set_is_ignored(self) -> None:
        """Nodes that leave the training set are not checked or warned about."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        checker.check(subsystem_name="training", subsystem_node_ids=set(), registered_agent_node_ids=set())
        checker.check(subsystem_name="training", subsystem_node_ids=set(), registered_agent_node_ids=set())


class TestCoverageResultStructured:
    """Previously check() only logged warnings and returned None. Now it returns
    CoverageResult with persistently_uncovered_node_ids, enabling tick_loop to
    escalate to formal notification instead of just logging."""

    def test_returns_persistently_uncovered_at_threshold(self) -> None:
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=3)

        for _ in range(2):
            result = checker.check(
                subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set()
            )
            assert result.persistently_uncovered_node_ids == []

        result = checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        assert result.persistently_uncovered_node_ids == ["n1"]

    def test_returns_empty_when_all_covered(self) -> None:
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        result = checker.check(
            subsystem_name="training", subsystem_node_ids={"n1", "n2"}, registered_agent_node_ids={"n1", "n2"}
        )

        assert result.persistently_uncovered_node_ids == []
        assert result.newly_restored_node_ids == []

    def test_returns_newly_restored_on_recovery(self) -> None:
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        result = checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids={"n1"})

        assert result.newly_restored_node_ids == ["n1"]

    def test_not_reported_again_until_restored(self) -> None:
        """After threshold is reached, node is not re-reported in subsequent calls."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        result1 = checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        result2 = checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())

        assert result1.persistently_uncovered_node_ids == ["n1"]
        assert result2.persistently_uncovered_node_ids == []

    def test_result_carries_subsystem_name(self) -> None:
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        result = checker.check(
            subsystem_name="rollout_default", subsystem_node_ids=set(), registered_agent_node_ids=set()
        )

        assert result.subsystem_name == "rollout_default"


class TestCoveragePerSubsystem:
    """Previously coverage checker only tracked training nodes. Now it keys
    counters by (subsystem_name, node_id) so different subsystems are tracked
    independently."""

    def test_training_and_rollout_counted_independently(self) -> None:
        """Same physical node missing in both training and rollout triggers
        independent coverage alerts per subsystem."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        # Training: node n1 uncovered for 2 ticks → triggers
        checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        r_train = checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())

        # Rollout: node n1 uncovered for only 1 tick → does not trigger yet
        r_rollout = checker.check(
            subsystem_name="rollout_default", subsystem_node_ids={"n1"}, registered_agent_node_ids=set()
        )

        assert r_train.persistently_uncovered_node_ids == ["n1"]
        assert r_train.subsystem_name == "training"
        assert r_rollout.persistently_uncovered_node_ids == []
        assert r_rollout.subsystem_name == "rollout_default"

    def test_rollout_coverage_gap_triggers_alert(self) -> None:
        """Rollout nodes missing agent coverage should trigger alerts.
        Previously, rollout nodes were not checked at all."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        checker.check(subsystem_name="rollout_0", subsystem_node_ids={"r1", "r2"}, registered_agent_node_ids=set())
        result = checker.check(
            subsystem_name="rollout_0", subsystem_node_ids={"r1", "r2"}, registered_agent_node_ids=set()
        )

        assert sorted(result.persistently_uncovered_node_ids) == ["r1", "r2"]
        assert result.subsystem_name == "rollout_0"

    def test_rollout_coverage_restored_clears_state(self) -> None:
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        checker.check(subsystem_name="rollout_0", subsystem_node_ids={"r1"}, registered_agent_node_ids=set())
        checker.check(subsystem_name="rollout_0", subsystem_node_ids={"r1"}, registered_agent_node_ids=set())
        result = checker.check(subsystem_name="rollout_0", subsystem_node_ids={"r1"}, registered_agent_node_ids={"r1"})

        assert result.newly_restored_node_ids == ["r1"]

    def test_same_node_in_different_subsystems_tracked_separately(self) -> None:
        """A physical node shared between training and rollout should have
        independent counters for each subsystem."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        # Node n1 covered in training but uncovered in rollout
        checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids={"n1"})
        checker.check(subsystem_name="rollout_0", subsystem_node_ids={"n1"}, registered_agent_node_ids=set())

        r_train = checker.check(subsystem_name="training", subsystem_node_ids={"n1"}, registered_agent_node_ids={"n1"})
        r_rollout = checker.check(
            subsystem_name="rollout_0", subsystem_node_ids={"n1"}, registered_agent_node_ids=set()
        )

        assert r_train.persistently_uncovered_node_ids == []
        assert r_rollout.persistently_uncovered_node_ids == ["n1"]

    def test_no_stale_uncovered_when_subsystem_has_no_active_nodes(self) -> None:
        """When a rollout subsystem reports no active nodes, its previous
        uncovered state should not linger."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        checker.check(subsystem_name="rollout_0", subsystem_node_ids={"r1"}, registered_agent_node_ids=set())
        checker.check(subsystem_name="rollout_0", subsystem_node_ids={"r1"}, registered_agent_node_ids=set())

        # Subsystem scales down — no active nodes
        result = checker.check(subsystem_name="rollout_0", subsystem_node_ids=set(), registered_agent_node_ids=set())

        assert result.persistently_uncovered_node_ids == []
