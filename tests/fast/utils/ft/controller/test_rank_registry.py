"""Tests for RankRegistry."""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.rank_registry import RankRegistry


class _FakeScrapeTargetManager:
    """Records add/remove calls for assertion."""

    def __init__(self) -> None:
        self.targets: dict[str, str] = {}

    def add_scrape_target(self, target_id: str, address: str) -> None:
        self.targets[target_id] = address

    def remove_scrape_target(self, target_id: str) -> None:
        self.targets.pop(target_id, None)


def _make_registry(
    *,
    run_id: str | None = "run-1",
    scrape_target_manager: _FakeScrapeTargetManager | None = None,
) -> RankRegistry:
    return RankRegistry(
        run_id=run_id,
        scrape_target_manager=scrape_target_manager,
    )


_VALID_KWARGS: dict = dict(
    run_id="run-1",
    rank=0,
    world_size=2,
    node_id="node-0",
    exporter_address="http://node-0:9090",
)


# ===================================================================
# Validation
# ===================================================================


class TestRegisterTrainingRankValidation:
    def test_empty_node_id_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(ValueError, match="node_id must be non-empty"):
            registry.register_training_rank(**{**_VALID_KWARGS, "node_id": ""})

    def test_zero_world_size_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(ValueError, match="world_size must be positive"):
            registry.register_training_rank(**{**_VALID_KWARGS, "world_size": 0})

    def test_negative_world_size_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(ValueError, match="world_size must be positive"):
            registry.register_training_rank(**{**_VALID_KWARGS, "world_size": -1})

    def test_negative_rank_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(ValueError, match=r"rank must be in \[0, 2\)"):
            registry.register_training_rank(**{**_VALID_KWARGS, "rank": -1})

    def test_rank_equal_to_world_size_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(ValueError, match=r"rank must be in \[0, 2\)"):
            registry.register_training_rank(**{**_VALID_KWARGS, "rank": 2})

    def test_rank_exceeding_world_size_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(ValueError, match=r"rank must be in \[0, 2\)"):
            registry.register_training_rank(**{**_VALID_KWARGS, "rank": 5})


# ===================================================================
# run_id matching
# ===================================================================


class TestRunIdMatching:
    def test_matching_run_id_accepted(self) -> None:
        registry = _make_registry(run_id="run-1")
        registry.register_training_rank(**_VALID_KWARGS, pid=42)

        assert registry.rank_placement == {0: "node-0"}
        assert registry.rank_pids == {0: 42}

    def test_mismatched_run_id_rejected(self) -> None:
        registry = _make_registry(run_id="run-1")
        registry.register_training_rank(**{**_VALID_KWARGS, "run_id": "run-other"})

        assert registry.rank_placement == {}

    def test_none_run_id_rejects_everything(self) -> None:
        registry = _make_registry(run_id=None)
        registry.register_training_rank(**_VALID_KWARGS)

        assert registry.rank_placement == {}


class TestRegisterTrainingRankStalePid:
    """Verify that re-registering a rank without pid does not retain the old pid."""

    @pytest.mark.xfail(
        reason="Known issue: rank_pids not cleared when re-registering without pid on new node",
        strict=True,
    )
    def test_reregister_same_run_without_pid_clears_old_pid(self) -> None:
        registry = _make_registry()
        registry.register_training_rank(**_VALID_KWARGS, pid=1234)
        assert registry.rank_pids == {0: 1234}

        registry.register_training_rank(**{**_VALID_KWARGS, "node_id": "node-0-new"})
        assert 0 not in registry.rank_pids, (
            "Old pid should be cleared when rank re-registers without pid"
        )


# ===================================================================
# register_training_rank happy path
# ===================================================================


class TestRegisterTrainingRankHappyPath:
    def test_state_after_single_registration(self) -> None:
        registry = _make_registry()
        registry.register_training_rank(**_VALID_KWARGS, pid=42)

        assert registry.rank_placement == {0: "node-0"}
        assert registry.expected_world_size == 2
        assert registry.rank_pids == {0: 42}
        assert registry.run_id == "run-1"

    def test_two_ranks_same_run(self) -> None:
        registry = _make_registry()

        registry.register_training_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090", pid=10,
        )
        registry.register_training_rank(
            run_id="run-1", rank=1, world_size=2,
            node_id="node-1", exporter_address="http://node-1:9090", pid=20,
        )

        assert registry.rank_placement == {0: "node-0", 1: "node-1"}
        assert registry.rank_pids == {0: 10, 1: 20}
        assert registry.expected_world_size == 2

    def test_pid_none_not_recorded(self) -> None:
        registry = _make_registry()
        registry.register_training_rank(**_VALID_KWARGS)

        assert 0 not in registry.rank_pids
        assert registry.rank_placement == {0: "node-0"}


# ===================================================================
# scrape target management
# ===================================================================


class TestScrapeTargetManager:
    def test_register_adds_scrape_target(self) -> None:
        stm = _FakeScrapeTargetManager()
        registry = _make_registry(scrape_target_manager=stm)

        registry.register_training_rank(**_VALID_KWARGS)

        assert stm.targets == {"rank-0": "http://node-0:9090"}

    def test_no_scrape_target_manager_is_safe(self) -> None:
        registry = _make_registry(scrape_target_manager=None)
        registry.register_training_rank(**_VALID_KWARGS)

    def test_cleanup_removes_all_scrape_targets(self) -> None:
        stm = _FakeScrapeTargetManager()
        registry = _make_registry(scrape_target_manager=stm)

        registry.register_training_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        registry.register_training_rank(
            run_id="run-1", rank=1, world_size=2,
            node_id="node-1", exporter_address="http://node-1:9090",
        )
        assert len(stm.targets) == 2

        registry.cleanup()

        assert stm.targets == {}

    def test_cleanup_without_scrape_target_manager_is_safe(self) -> None:
        registry = _make_registry(scrape_target_manager=None)
        registry.register_training_rank(**_VALID_KWARGS)
        registry.cleanup()


# ===================================================================
# get_rank_pids_for_node
# ===================================================================


class TestGetRankPidsForNode:
    def test_returns_matching_ranks(self) -> None:
        registry = _make_registry()
        registry.register_training_rank(
            run_id="run-1", rank=0, world_size=4,
            node_id="node-A", exporter_address="addr", pid=10,
        )
        registry.register_training_rank(
            run_id="run-1", rank=1, world_size=4,
            node_id="node-A", exporter_address="addr", pid=20,
        )
        registry.register_training_rank(
            run_id="run-1", rank=2, world_size=4,
            node_id="node-B", exporter_address="addr", pid=30,
        )

        result = registry.get_rank_pids_for_node("node-A")

        assert result == {0: 10, 1: 20}

    def test_no_ranks_returns_empty(self) -> None:
        registry = _make_registry()

        assert registry.get_rank_pids_for_node("node-X") == {}

    def test_ranks_without_pids_excluded(self) -> None:
        registry = _make_registry()
        registry.register_training_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-A", exporter_address="addr",
        )
        registry.register_training_rank(
            run_id="run-1", rank=1, world_size=2,
            node_id="node-A", exporter_address="addr", pid=42,
        )

        result = registry.get_rank_pids_for_node("node-A")

        assert result == {1: 42}
