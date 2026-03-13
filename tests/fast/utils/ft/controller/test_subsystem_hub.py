"""Unit tests for SubsystemHub (P0 item 2)."""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.subsystem_hub.hub import SubsystemHub
from miles.utils.ft.controller.subsystem_hub.training_rank_roster import TrainingRankRoster
from miles.utils.ft.controller.types import NullScrapeTargetManager
from miles.utils.ft.utils.box import Box


class TestTrainingRankRosterProperty:
    def test_raises_when_box_is_none(self) -> None:
        hub = SubsystemHub(training_rank_roster_box=Box(None))
        with pytest.raises(AssertionError, match="not yet initialized"):
            _ = hub.training_rank_roster

    def test_returns_roster_when_set(self) -> None:
        roster = TrainingRankRoster(
            run_id="test-run",
            scrape_target_manager=NullScrapeTargetManager(),
        )
        hub = SubsystemHub(training_rank_roster_box=Box(roster))
        assert hub.training_rank_roster is roster

    def test_training_rank_roster_box_property(self) -> None:
        box: Box[TrainingRankRoster | None] = Box(None)
        hub = SubsystemHub(training_rank_roster_box=box)
        assert hub.training_rank_roster_box is box


class TestRolloutManagerHandle:
    def test_raises_when_not_set(self) -> None:
        hub = SubsystemHub(training_rank_roster_box=Box(None))
        with pytest.raises(AssertionError, match="not yet set"):
            _ = hub.rollout_manager_handle

    def test_returns_handle_after_set(self) -> None:
        hub = SubsystemHub(training_rank_roster_box=Box(None))
        handle = object()
        hub.set_rollout_handle(handle)
        assert hub.rollout_manager_handle is handle


class TestRolloutNodeIds:
    def test_get_returns_empty_frozenset_for_unknown_cell(self) -> None:
        hub = SubsystemHub(training_rank_roster_box=Box(None))
        result = hub.get_rollout_node_ids("cell-0")
        assert result == frozenset()
        assert isinstance(result, frozenset)

    def test_set_and_get_round_trip(self) -> None:
        hub = SubsystemHub(training_rank_roster_box=Box(None))
        hub.set_rollout_node_ids("cell-0", {"node-0", "node-1"})
        result = hub.get_rollout_node_ids("cell-0")
        assert result == frozenset({"node-0", "node-1"})
        assert isinstance(result, frozenset)

    def test_per_cell_isolation(self) -> None:
        hub = SubsystemHub(training_rank_roster_box=Box(None))
        hub.set_rollout_node_ids("cell-a", {"node-0"})
        hub.set_rollout_node_ids("cell-b", {"node-1", "node-2"})

        assert hub.get_rollout_node_ids("cell-a") == frozenset({"node-0"})
        assert hub.get_rollout_node_ids("cell-b") == frozenset({"node-1", "node-2"})

    def test_stored_as_frozenset_cuts_external_alias(self) -> None:
        """set_rollout_node_ids stores a frozenset copy, so mutating the
        original mutable set cannot affect the hub's internal state."""
        hub = SubsystemHub(training_rank_roster_box=Box(None))
        mutable: set[str] = {"node-0"}
        hub.set_rollout_node_ids("cell-0", mutable)

        mutable.add("node-injected")

        assert hub.get_rollout_node_ids("cell-0") == frozenset({"node-0"})

    def test_rollout_node_ids_empty_means_detectors_wont_run(self) -> None:
        """Without set_rollout_node_ids, get_rollout_node_ids returns empty,
        which causes build_subsystem_context to set should_run_detectors=False.
        Previously register_rollout() never called set_rollout_node_ids,
        so rollout detectors were permanently disabled."""
        hub = SubsystemHub(training_rank_roster_box=Box(None))

        hub.set_rollout_handle(object())
        assert hub.get_rollout_node_ids("cell-0") == frozenset()

        hub.set_rollout_node_ids("cell-0", ["node-a", "node-b"])
        assert hub.get_rollout_node_ids("cell-0") == frozenset({"node-a", "node-b"})
