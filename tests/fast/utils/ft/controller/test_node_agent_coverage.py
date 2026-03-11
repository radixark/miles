from __future__ import annotations

import logging

import pytest

from miles.utils.ft.controller.node_agent_coverage import NodeAgentCoverageChecker


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

    def test_node_not_in_training_set_is_ignored(self) -> None:
        """Nodes that leave the training set are not checked or warned about."""
        checker = NodeAgentCoverageChecker(window_seconds=600, threshold=2)

        checker.check(subsystem_node_ids={"n1"}, registered_agent_node_ids=set())
        checker.check(subsystem_node_ids=set(), registered_agent_node_ids=set())
        checker.check(subsystem_node_ids=set(), registered_agent_node_ids=set())
