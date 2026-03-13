from __future__ import annotations

import logging

import pytest

from miles.utils.ft.controller.node_agents import CoverageResult, NodeAgentCoverageChecker


class TestNodeAgentCoverageChecker:
    def test_no_warning_below_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        """Uncovered nodes below threshold do not trigger a warning."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=3)

        with caplog.at_level(logging.WARNING):
            checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
            checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())

        assert "without node agent" not in caplog.text

    def test_warning_at_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning fires exactly when the uncovered count reaches the threshold."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=3)

        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())

        assert "without node agent" in caplog.text
        assert "n1" in caplog.text

    def test_no_duplicate_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Once warned, the same node does not trigger repeated warnings."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())

        warning_count = caplog.text.count("without node agent")
        assert warning_count == 1

    def test_coverage_restored_clears_alert(self, caplog: pytest.LogCaptureFixture) -> None:
        """Restoring coverage logs info and allows re-alerting if coverage drops again."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        with caplog.at_level(logging.INFO):
            checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
            checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
            checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids={"n1"})

        assert "coverage restored" in caplog.text

    def test_realert_after_restore_and_drop(self, caplog: pytest.LogCaptureFixture) -> None:
        """After coverage is restored and drops again, a new warning fires."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        with caplog.at_level(logging.WARNING):
            checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
            checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
            checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids={"n1"})
            checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
            checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())

        warning_count = caplog.text.count("without node agent")
        assert warning_count == 2

    def test_multiple_nodes_tracked_independently(self, caplog: pytest.LogCaptureFixture) -> None:
        """Each node's uncovered window is independent."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        with caplog.at_level(logging.WARNING):
            checker.check(subsystem_node_ids={"n1", "n2"}, registered_agent_node_ids={"n2"})
            checker.check(subsystem_node_ids={"n1", "n2"}, registered_agent_node_ids={"n2"})

        warning_lines = [r.message for r in caplog.records if "without node agent" in r.message]
        assert len(warning_lines) == 1
        assert "n1" in warning_lines[0]

    # P2 item 26: oscillating coverage (covered → uncovered → covered → uncovered)
    def test_oscillating_coverage_rearms_notification_each_cycle(self, caplog: pytest.LogCaptureFixture) -> None:
        """Three full oscillation cycles: each drop after restore triggers a new warning."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
                checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
                checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids={"n1"})

        warning_count = caplog.text.count("without node agent")
        assert warning_count == 3

    def test_node_not_in_training_set_is_ignored(self) -> None:
        """Nodes that leave the training set are not checked or warned about."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        checker.check(subsystem_node_ids=set(), registered_agent_node_ids=set())
        checker.check(subsystem_node_ids=set(), registered_agent_node_ids=set())


class TestCoverageResultStructured:
    """Previously check() only logged warnings and returned None. Now it returns
    CoverageResult with persistently_uncovered_node_ids, enabling tick_loop to
    escalate to formal notification instead of just logging."""

    def test_returns_persistently_uncovered_at_threshold(self) -> None:
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=3)

        for _ in range(2):
            result = checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
            assert result.persistently_uncovered_node_ids == []

        result = checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        assert result.persistently_uncovered_node_ids == ["n1"]

    def test_returns_empty_when_all_covered(self) -> None:
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        result = checker.check(subsystem_node_ids={"n1", "n2"}, registered_agent_node_ids={"n1", "n2"})

        assert result.persistently_uncovered_node_ids == []
        assert result.newly_restored_node_ids == []

    def test_returns_newly_restored_on_recovery(self) -> None:
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        result = checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids={"n1"})

        assert result.newly_restored_node_ids == ["n1"]

    def test_not_reported_again_until_restored(self) -> None:
        """After threshold is reached, node is not re-reported in subsequent calls."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        result1 = checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        result2 = checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())

        assert result1.persistently_uncovered_node_ids == ["n1"]
        assert result2.persistently_uncovered_node_ids == []
