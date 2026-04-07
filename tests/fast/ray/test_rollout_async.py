"""Tests for cell-based FT APIs on RolloutManager.

Verifies that:
- list_cells() enumerates all non-placeholder server groups
- start_cell() restarts dead engines via ServerGroup.start_engines
- stop_cell(cell_id, timeout_seconds) kills engines and sets slots to None
- get_cell_status() returns "running" / "stopped" / "failed"
- get_cell_node_ids() returns deduplicated sorted IPs
- _find_group_by_cell_id() raises ValueError for unknown cell

No GPU, no Ray cluster required — pure mock.
"""

from unittest.mock import MagicMock, patch

import pytest

from miles.ray.rollout import RolloutManager, RolloutServer, ServerGroup

_RawRolloutManager = RolloutManager.__ray_actor_class__


def _mock_group(cell_id: str, worker_type: str = "regular", num_engines: int = 2, dead_indices=None):
    dead_indices = set(dead_indices or [])
    engines = [MagicMock() if i not in dead_indices else None for i in range(num_engines)]

    group = MagicMock(spec=ServerGroup)
    group.cell_id = cell_id
    group.worker_type = worker_type
    group.all_engines = engines
    group.engines = [e for e in engines if e is not None]
    group.num_gpus_per_engine = 1
    group.needs_offload = False

    def fake_start_engines(port_cursors=None):
        handles = []
        for i in range(len(group.all_engines)):
            if group.all_engines[i] is None:
                group.all_engines[i] = MagicMock()
                handles.append(MagicMock())
        group.num_new_engines = len(handles)
        return handles, port_cursors or {}

    group.start_engines = fake_start_engines
    return group


def _mock_server(groups):
    srv = MagicMock(spec=RolloutServer)
    srv.server_groups = groups
    srv.engines = [e for g in groups for e in g.engines]
    srv.all_engines = [e for g in groups for e in g.all_engines]
    return srv


def _make_manager_with_servers(servers_dict):
    """Create a minimal RolloutManager-like object with .servers set, bypassing __init__."""
    mgr = object.__new__(_RawRolloutManager)
    mgr.servers = servers_dict
    return mgr


class TestListCells:
    def test_enumerates_all_groups(self):
        g1 = _mock_group("default-regular")
        g2 = _mock_group("default-prefill", worker_type="prefill")
        srv = _mock_server([g1, g2])
        mgr = _make_manager_with_servers({"default": srv})

        cells = mgr.list_cells()
        assert cells == [{"cell_id": "default-regular"}, {"cell_id": "default-prefill"}]

    def test_skips_placeholder_groups(self):
        g1 = _mock_group("default-regular")
        g2 = _mock_group("", worker_type="placeholder")
        srv = _mock_server([g1, g2])
        mgr = _make_manager_with_servers({"default": srv})

        cells = mgr.list_cells()
        assert cells == [{"cell_id": "default-regular"}]

    def test_multi_model(self):
        g1 = _mock_group("actor-regular")
        g2 = _mock_group("ref-regular")
        srv1 = _mock_server([g1])
        srv2 = _mock_server([g2])
        mgr = _make_manager_with_servers({"actor": srv1, "ref": srv2})

        cells = mgr.list_cells()
        ids = [c["cell_id"] for c in cells]
        assert "actor-regular" in ids
        assert "ref-regular" in ids

    def test_empty_servers(self):
        mgr = _make_manager_with_servers({})
        assert mgr.list_cells() == []


class TestFindGroupByCellId:
    def test_finds_existing_cell(self):
        g = _mock_group("default-regular")
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        assert mgr._find_group_by_cell_id("default-regular") is g

    def test_raises_on_unknown_cell(self):
        g = _mock_group("default-regular")
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        with pytest.raises(ValueError, match="not found"):
            mgr._find_group_by_cell_id("nonexistent-cell")


class TestStartCell:
    @patch("miles.ray.rollout.ray")
    def test_restarts_dead_engines(self, mock_ray):
        mock_ray.get = MagicMock(return_value=None)
        g = _mock_group("default-regular", num_engines=2, dead_indices=[0, 1])
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        num = mgr.start_cell("default-regular")
        assert num == 2
        assert all(e is not None for e in g.all_engines)
        mock_ray.get.assert_called_once()

    @patch("miles.ray.rollout.ray")
    def test_noop_when_all_healthy(self, mock_ray):
        g = _mock_group("default-regular", num_engines=2, dead_indices=[])
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        num = mgr.start_cell("default-regular")
        assert num == 0
        mock_ray.get.assert_not_called()


class TestStopCell:
    @patch("miles.ray.rollout.ray")
    def test_kills_all_engines(self, mock_ray):
        g = _mock_group("default-regular", num_engines=2)
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        mgr.stop_cell("default-regular")
        assert all(e is None for e in g.all_engines)
        assert mock_ray.kill.call_count == 2

    @patch("miles.ray.rollout.ray")
    def test_with_timeout(self, mock_ray):
        g = _mock_group("default-regular", num_engines=1)
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        mgr.stop_cell("default-regular", timeout_seconds=60)
        assert g.all_engines[0] is None

    @patch("miles.ray.rollout.ray")
    def test_kill_exception_still_sets_none(self, mock_ray):
        mock_ray.kill.side_effect = RuntimeError("actor already dead")
        g = _mock_group("default-regular", num_engines=1)
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        mgr.stop_cell("default-regular")
        assert g.all_engines[0] is None


class TestGetCellStatus:
    def test_running(self):
        g = _mock_group("default-regular", num_engines=2)
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        assert mgr.get_cell_status("default-regular") == "running"

    def test_stopped(self):
        g = _mock_group("default-regular", num_engines=2, dead_indices=[0, 1])
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        assert mgr.get_cell_status("default-regular") == "stopped"

    def test_failed_partial(self):
        g = _mock_group("default-regular", num_engines=2, dead_indices=[1])
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        assert mgr.get_cell_status("default-regular") == "failed"

    def test_empty_engines(self):
        g = _mock_group("default-regular", num_engines=0)
        g.all_engines = []
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        assert mgr.get_cell_status("default-regular") == "stopped"


class TestGetCellNodeIds:
    @patch("miles.ray.rollout.ray")
    def test_returns_sorted_unique_ips(self, mock_ray):
        g = _mock_group("default-regular", num_engines=3)
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        mock_ray.get.return_value = [
            ("10.0.0.2", 15000),
            ("10.0.0.1", 15001),
            ("10.0.0.2", 15002),
        ]

        ids = mgr.get_cell_node_ids("default-regular")
        assert ids == ["10.0.0.1", "10.0.0.2"]

    @patch("miles.ray.rollout.ray")
    def test_skips_dead_engines(self, mock_ray):
        g = _mock_group("default-regular", num_engines=2, dead_indices=[1])
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        mock_ray.get.return_value = [("10.0.0.1", 15000)]

        ids = mgr.get_cell_node_ids("default-regular")
        assert ids == ["10.0.0.1"]

    def test_all_dead_returns_empty(self):
        g = _mock_group("default-regular", num_engines=2, dead_indices=[0, 1])
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        ids = mgr.get_cell_node_ids("default-regular")
        assert ids == []

    @patch("miles.ray.rollout.ray")
    def test_exception_returns_empty(self, mock_ray):
        mock_ray.get.side_effect = RuntimeError("network error")
        g = _mock_group("default-regular", num_engines=1)
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        ids = mgr.get_cell_node_ids("default-regular")
        assert ids == []


class TestGpuFailureRecoveryFlow:
    """End-to-end flow tests simulating GPU failure → detection → recovery."""

    @patch("miles.ray.rollout.ray")
    def test_single_engine_dies_and_recovers(self, mock_ray):
        """One of four engines dies → status becomes 'failed' → stop → start → back to 'running'."""
        mock_ray.get = MagicMock(return_value=None)
        mock_ray.kill = MagicMock()
        g = _mock_group("default-regular", num_engines=4)
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        assert mgr.get_cell_status("default-regular") == "running"

        # Simulate engine 2 crash (health monitor would set this to None)
        g.all_engines[2] = None
        assert mgr.get_cell_status("default-regular") == "failed"

        # FT controller calls stop_cell to clean up remaining engines
        mgr.stop_cell("default-regular")
        assert mgr.get_cell_status("default-regular") == "stopped"
        assert all(e is None for e in g.all_engines)

        # FT controller calls start_cell to restart all engines
        num_restarted = mgr.start_cell("default-regular")
        assert num_restarted == 4
        assert mgr.get_cell_status("default-regular") == "running"
        assert all(e is not None for e in g.all_engines)

    @patch("miles.ray.rollout.ray")
    def test_all_engines_die_and_recover(self, mock_ray):
        """All engines crash → status 'stopped' → start brings them back."""
        mock_ray.get = MagicMock(return_value=None)
        g = _mock_group("default-regular", num_engines=3, dead_indices=[0, 1, 2])
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        assert mgr.get_cell_status("default-regular") == "stopped"

        num_restarted = mgr.start_cell("default-regular")
        assert num_restarted == 3
        assert mgr.get_cell_status("default-regular") == "running"

    @patch("miles.ray.rollout.ray")
    def test_partial_failure_direct_restart(self, mock_ray):
        """One engine dies → start_cell only restarts the dead one (without stop first)."""
        mock_ray.get = MagicMock(return_value=None)
        g = _mock_group("default-regular", num_engines=4)
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        original_engine_0 = g.all_engines[0]
        original_engine_1 = g.all_engines[1]

        # Simulate engine 3 crash
        g.all_engines[3] = None
        assert mgr.get_cell_status("default-regular") == "failed"

        # start_cell only restarts the dead slot, healthy engines stay
        num_restarted = mgr.start_cell("default-regular")
        assert num_restarted == 1
        assert g.all_engines[0] is original_engine_0
        assert g.all_engines[1] is original_engine_1
        assert g.all_engines[3] is not None
        assert mgr.get_cell_status("default-regular") == "running"

    @patch("miles.ray.rollout.ray")
    def test_multi_model_independent_failure(self, mock_ray):
        """GPU failure on one model doesn't affect the other model's cell status."""
        mock_ray.get = MagicMock(return_value=None)
        mock_ray.kill = MagicMock()
        g_actor = _mock_group("actor-regular", num_engines=2)
        g_ref = _mock_group("ref-regular", num_engines=2)
        srv_actor = _mock_server([g_actor])
        srv_ref = _mock_server([g_ref])
        mgr = _make_manager_with_servers({"actor": srv_actor, "ref": srv_ref})

        assert mgr.get_cell_status("actor-regular") == "running"
        assert mgr.get_cell_status("ref-regular") == "running"

        # Actor model engine 0 crashes
        g_actor.all_engines[0] = None
        assert mgr.get_cell_status("actor-regular") == "failed"
        assert mgr.get_cell_status("ref-regular") == "running"

        # Recover actor model only
        mgr.stop_cell("actor-regular")
        mgr.start_cell("actor-regular")
        assert mgr.get_cell_status("actor-regular") == "running"
        assert mgr.get_cell_status("ref-regular") == "running"

    @patch("miles.ray.rollout.ray")
    def test_repeated_failure_and_recovery(self, mock_ray):
        """Engine fails twice in a row — system recovers each time."""
        mock_ray.get = MagicMock(return_value=None)
        mock_ray.kill = MagicMock()
        g = _mock_group("default-regular", num_engines=2)
        srv = _mock_server([g])
        mgr = _make_manager_with_servers({"default": srv})

        for _ in range(2):
            assert mgr.get_cell_status("default-regular") == "running"

            g.all_engines[0] = None
            assert mgr.get_cell_status("default-regular") == "failed"

            mgr.stop_cell("default-regular")
            assert mgr.get_cell_status("default-regular") == "stopped"

            mgr.start_cell("default-regular")
            assert mgr.get_cell_status("default-regular") == "running"
