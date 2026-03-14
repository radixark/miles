"""Tests for TrainingRankRoster."""

from __future__ import annotations

import pytest

from miles.utils.ft.controller.subsystem_hub import TrainingRankRoster
from miles.utils.ft.controller.types import NullScrapeTargetManager


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
    run_id: str = "run-1",
    scrape_target_manager: _FakeScrapeTargetManager | NullScrapeTargetManager | None = None,
) -> TrainingRankRoster:
    return TrainingRankRoster(
        run_id=run_id,
        scrape_target_manager=scrape_target_manager or NullScrapeTargetManager(),
    )


_VALID_KWARGS: dict = dict(
    run_id="run-1",
    rank=0,
    world_size=2,
    node_id="node-0",
    exporter_address="http://node-0:9090",
    pid=1,
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
        registry.register_training_rank(**{**_VALID_KWARGS, "pid": 42})

        assert registry.rank_placement == {0: "node-0"}
        assert registry.rank_pids == {0: 42}

    def test_mismatched_run_id_rejected(self) -> None:
        registry = _make_registry(run_id="run-1")
        registry.register_training_rank(**{**_VALID_KWARGS, "run_id": "run-other"})

        assert registry.rank_placement == {}

    def test_empty_run_id_rejects_real_run_id(self) -> None:
        registry = _make_registry(run_id="")
        registry.register_training_rank(**_VALID_KWARGS)

        assert registry.rank_placement == {}


# ===================================================================
# register_training_rank happy path
# ===================================================================


class TestRegisterTrainingRankHappyPath:
    def test_state_after_single_registration(self) -> None:
        registry = _make_registry()
        registry.register_training_rank(**{**_VALID_KWARGS, "pid": 42})

        assert registry.rank_placement == {0: "node-0"}
        assert registry.expected_world_size == 2
        assert registry.rank_pids == {0: 42}
        assert registry.run_id == "run-1"

    def test_two_ranks_same_run(self) -> None:
        registry = _make_registry()

        registry.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=10,
        )
        registry.register_training_rank(
            run_id="run-1",
            rank=1,
            world_size=2,
            node_id="node-1",
            exporter_address="http://node-1:9090",
            pid=20,
        )

        assert registry.rank_placement == {0: "node-0", 1: "node-1"}
        assert registry.rank_pids == {0: 10, 1: 20}
        assert registry.expected_world_size == 2

    def test_inconsistent_world_size_rejected(self, caplog: pytest.LogCaptureFixture) -> None:
        """world_size was overwritten on every registration, allowing
        inconsistent values to drift the completeness check. Now locked
        after first registration; mismatches are rejected."""
        registry = _make_registry()

        registry.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=4,
            node_id="node-0",
            exporter_address="addr",
            pid=10,
        )
        assert registry.expected_world_size == 4

        with caplog.at_level("ERROR"):
            registry.register_training_rank(
                run_id="run-1",
                rank=1,
                world_size=8,
                node_id="node-1",
                exporter_address="addr",
                pid=20,
            )

        assert "rejected inconsistent world_size" in caplog.text
        assert registry.expected_world_size == 4
        assert 1 not in registry.rank_placement
        assert 1 not in registry.rank_pids


# ===================================================================
# scrape target management
# ===================================================================


class TestScrapeTargetManager:
    def test_register_adds_scrape_target(self) -> None:
        stm = _FakeScrapeTargetManager()
        registry = _make_registry(scrape_target_manager=stm)

        registry.register_training_rank(**_VALID_KWARGS)

        assert stm.targets == {"rank-0": "http://node-0:9090"}

    def test_null_scrape_target_manager_is_safe(self) -> None:
        registry = _make_registry(scrape_target_manager=NullScrapeTargetManager())
        registry.register_training_rank(**_VALID_KWARGS)

    def test_cleanup_removes_all_scrape_targets(self) -> None:
        stm = _FakeScrapeTargetManager()
        registry = _make_registry(scrape_target_manager=stm)

        registry.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9090",
            pid=10,
        )
        registry.register_training_rank(
            run_id="run-1",
            rank=1,
            world_size=2,
            node_id="node-1",
            exporter_address="http://node-1:9090",
            pid=20,
        )
        assert len(stm.targets) == 2

        registry.cleanup()

        assert stm.targets == {}

    def test_cleanup_with_null_scrape_target_manager_is_safe(self) -> None:
        registry = _make_registry(scrape_target_manager=NullScrapeTargetManager())
        registry.register_training_rank(**_VALID_KWARGS)
        registry.cleanup()


# ===================================================================
# get_rank_pids_for_node
# ===================================================================


class TestGetRankPidsForNode:
    def test_returns_matching_ranks(self) -> None:
        registry = _make_registry()
        registry.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=4,
            node_id="node-A",
            exporter_address="addr",
            pid=10,
        )
        registry.register_training_rank(
            run_id="run-1",
            rank=1,
            world_size=4,
            node_id="node-A",
            exporter_address="addr",
            pid=20,
        )
        registry.register_training_rank(
            run_id="run-1",
            rank=2,
            world_size=4,
            node_id="node-B",
            exporter_address="addr",
            pid=30,
        )

        result = registry.get_rank_pids_for_node("node-A")

        assert result == {0: 10, 1: 20}

    def test_no_ranks_returns_empty(self) -> None:
        registry = _make_registry()

        assert registry.get_rank_pids_for_node("node-X") == {}


# ===================================================================
# warn_if_incomplete
# ===================================================================


class TestWarnIfIncomplete:
    def test_warns_when_registered_less_than_expected(self, caplog: pytest.LogCaptureFixture) -> None:
        registry = _make_registry()
        registry.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=4,
            node_id="node-0",
            exporter_address="addr",
            pid=10,
        )
        registry.register_training_rank(
            run_id="run-1",
            rank=1,
            world_size=4,
            node_id="node-1",
            exporter_address="addr",
            pid=20,
        )

        with caplog.at_level("WARNING"):
            registry.warn_if_incomplete()

        assert "incomplete rank registration" in caplog.text
        assert "registered=2" in caplog.text
        assert "expected=4" in caplog.text

    def test_no_warn_when_expected_world_size_is_none(self, caplog: pytest.LogCaptureFixture) -> None:
        registry = _make_registry()

        with caplog.at_level("WARNING"):
            registry.warn_if_incomplete()

        assert "incomplete rank registration" not in caplog.text

    def test_no_warn_when_all_ranks_registered(self, caplog: pytest.LogCaptureFixture) -> None:
        registry = _make_registry()
        registry.register_training_rank(
            run_id="run-1",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="addr",
            pid=10,
        )
        registry.register_training_rank(
            run_id="run-1",
            rank=1,
            world_size=2,
            node_id="node-1",
            exporter_address="addr",
            pid=20,
        )

        with caplog.at_level("WARNING"):
            registry.warn_if_incomplete()

        assert "incomplete rank registration" not in caplog.text

    def test_no_warn_when_empty_roster(self, caplog: pytest.LogCaptureFixture) -> None:
        registry = _make_registry(run_id="unused")

        with caplog.at_level("WARNING"):
            registry.warn_if_incomplete()

        assert "incomplete rank registration" not in caplog.text
