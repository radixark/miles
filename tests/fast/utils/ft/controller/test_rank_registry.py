"""Tests for RankRegistry.register_rank() input validation."""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.rank_registry import RankRegistry


def _make_registry() -> RankRegistry:
    return RankRegistry(mini_wandb=MiniWandb())


_VALID_KWARGS: dict = dict(
    run_id="run-1",
    rank=0,
    world_size=2,
    node_id="node-0",
    exporter_address="http://node-0:9090",
)


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
