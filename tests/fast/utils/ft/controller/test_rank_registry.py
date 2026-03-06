"""Tests for RankRegistry."""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
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
    scrape_target_manager: _FakeScrapeTargetManager | None = None,
) -> RankRegistry:
    return RankRegistry(
        mini_wandb=MiniWandb(),
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
# Validation (existing coverage, kept for completeness)
# ===================================================================


class TestRegisterRankValidation:
    def test_empty_run_id_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(ValueError, match="run_id must be non-empty"):
            registry.register_rank(**{**_VALID_KWARGS, "run_id": ""})

    def test_empty_node_id_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(ValueError, match="node_id must be non-empty"):
            registry.register_rank(**{**_VALID_KWARGS, "node_id": ""})

    def test_zero_world_size_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(ValueError, match="world_size must be positive"):
            registry.register_rank(**{**_VALID_KWARGS, "world_size": 0})

    def test_negative_world_size_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(ValueError, match="world_size must be positive"):
            registry.register_rank(**{**_VALID_KWARGS, "world_size": -1})

    def test_negative_rank_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(ValueError, match=r"rank must be in \[0, 2\)"):
            registry.register_rank(**{**_VALID_KWARGS, "rank": -1})

    def test_rank_equal_to_world_size_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(ValueError, match=r"rank must be in \[0, 2\)"):
            registry.register_rank(**{**_VALID_KWARGS, "rank": 2})

    def test_rank_exceeding_world_size_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(ValueError, match=r"rank must be in \[0, 2\)"):
            registry.register_rank(**{**_VALID_KWARGS, "rank": 5})


class TestRegisterRankStalePid:
    """Verify that re-registering a rank without pid does not retain the old pid."""

    @pytest.mark.xfail(
        reason="Known issue: rank_pids not cleared when re-registering without pid on new node",
        strict=True,
    )
    def test_reregister_same_run_without_pid_clears_old_pid(self) -> None:
        registry = _make_registry()
        registry.register_rank(**_VALID_KWARGS, pid=1234)
        assert registry.rank_pids == {0: 1234}

        registry.register_rank(**{**_VALID_KWARGS, "node_id": "node-0-new"})
        assert 0 not in registry.rank_pids, (
            "Old pid should be cleared when rank re-registers without pid"
        )


# ===================================================================
# register_agent
# ===================================================================


class _FakeAgent:
    """Minimal stand-in for NodeAgentProtocol."""

    def __init__(self, name: str = "agent") -> None:
        self.name = name


class TestRegisterAgent:
    def test_stores_agent(self) -> None:
        registry = _make_registry()
        agent = _FakeAgent()

        registry.register_agent("node-0", agent)

        assert registry.agents["node-0"] is agent

    def test_overwrite_on_duplicate_node_id(self) -> None:
        registry = _make_registry()
        agent_old = _FakeAgent(name="old")
        agent_new = _FakeAgent(name="new")

        registry.register_agent("node-0", agent_old)
        registry.register_agent("node-0", agent_new)

        assert registry.agents["node-0"] is agent_new


# ===================================================================
# register_rank happy path
# ===================================================================


class TestRegisterRankHappyPath:
    def test_state_after_single_registration(self) -> None:
        registry = _make_registry()

        registry.register_rank(**_VALID_KWARGS, pid=42)

        assert registry.rank_placement == {0: "node-0"}
        assert registry.expected_world_size == 2
        assert registry.rank_pids == {0: 42}
        assert registry.active_run_id == "run-1"

    def test_two_ranks_same_run(self) -> None:
        registry = _make_registry()

        registry.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090", pid=10,
        )
        registry.register_rank(
            run_id="run-1", rank=1, world_size=2,
            node_id="node-1", exporter_address="http://node-1:9090", pid=20,
        )

        assert registry.rank_placement == {0: "node-0", 1: "node-1"}
        assert registry.rank_pids == {0: 10, 1: 20}
        assert registry.expected_world_size == 2

    def test_pid_none_not_recorded(self) -> None:
        registry = _make_registry()

        registry.register_rank(**_VALID_KWARGS)

        assert 0 not in registry.rank_pids
        assert registry.rank_placement == {0: "node-0"}


# ===================================================================
# _switch_run_if_needed
# ===================================================================


class TestSwitchRunIfNeeded:
    def test_same_run_id_no_switch(self) -> None:
        registry = _make_registry()
        registry.register_rank(**_VALID_KWARGS, pid=1)
        assert registry.active_run_id == "run-1"

        registry.register_rank(
            run_id="run-1", rank=1, world_size=2,
            node_id="node-1", exporter_address="http://node-1:9090", pid=2,
        )

        assert registry.rank_placement == {0: "node-0", 1: "node-1"}
        assert registry.rank_pids == {0: 1, 1: 2}

    def test_new_run_id_clears_old_state(self) -> None:
        registry = _make_registry()
        registry.register_rank(**_VALID_KWARGS, pid=1)
        registry._mini_wandb.log_step(run_id="run-1", step=1, metrics={"loss": 0.5})
        assert registry._mini_wandb.latest("loss") is not None

        registry.register_rank(
            run_id="run-2", rank=0, world_size=4,
            node_id="node-X", exporter_address="http://node-X:9090", pid=99,
        )

        assert registry.active_run_id == "run-2"
        assert registry.rank_placement == {0: "node-X"}
        assert registry.rank_pids == {0: 99}
        assert registry.expected_world_size == 4
        assert registry._mini_wandb.latest("loss") is None


# ===================================================================
# scrape target management
# ===================================================================


class TestScrapeTargetManager:
    def test_register_rank_adds_scrape_target(self) -> None:
        stm = _FakeScrapeTargetManager()
        registry = _make_registry(scrape_target_manager=stm)

        registry.register_rank(**_VALID_KWARGS)

        assert stm.targets == {"rank-0": "http://node-0:9090"}

    def test_no_scrape_target_manager_is_safe(self) -> None:
        registry = _make_registry(scrape_target_manager=None)
        registry.register_rank(**_VALID_KWARGS)

    def test_run_switch_removes_old_scrape_targets(self) -> None:
        stm = _FakeScrapeTargetManager()
        registry = _make_registry(scrape_target_manager=stm)

        registry.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        registry.register_rank(
            run_id="run-1", rank=1, world_size=2,
            node_id="node-1", exporter_address="http://node-1:9090",
        )
        assert len(stm.targets) == 2

        registry.register_rank(
            run_id="run-2", rank=0, world_size=2,
            node_id="node-A", exporter_address="http://node-A:9090",
        )

        assert "rank-1" not in stm.targets
        assert stm.targets == {"rank-0": "http://node-A:9090"}


# ===================================================================
# log_step
# ===================================================================


class TestLogStep:
    def test_delegates_to_mini_wandb(self) -> None:
        registry = _make_registry()
        registry.register_rank(**_VALID_KWARGS)

        registry.log_step(run_id="run-1", step=1, metrics={"loss": 0.5, "mfu": 0.4})

        assert registry.mini_wandb.latest("loss") == 0.5
        assert registry.mini_wandb.latest("mfu") == 0.4


# ===================================================================
# get_rank_pids_for_node
# ===================================================================


class TestGetRankPidsForNode:
    def test_returns_matching_ranks(self) -> None:
        registry = _make_registry()
        registry.register_rank(
            run_id="run-1", rank=0, world_size=4,
            node_id="node-A", exporter_address="addr", pid=10,
        )
        registry.register_rank(
            run_id="run-1", rank=1, world_size=4,
            node_id="node-A", exporter_address="addr", pid=20,
        )
        registry.register_rank(
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
        registry.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-A", exporter_address="addr",
        )
        registry.register_rank(
            run_id="run-1", rank=1, world_size=2,
            node_id="node-A", exporter_address="addr", pid=42,
        )

        result = registry.get_rank_pids_for_node("node-A")

        assert result == {1: 42}


# ===================================================================
# mini_wandb property
# ===================================================================


class TestMiniWandbProperty:
    def test_returns_injected_instance(self) -> None:
        wandb = MiniWandb()
        registry = RankRegistry(mini_wandb=wandb)

        assert registry.mini_wandb is wandb
