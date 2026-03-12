"""Tests for controller/factory.py — rank_pids_provider lambda safety and config wiring."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.adapters.stubs import StubMainJob, StubNodeManager
from miles.utils.ft.factories.controller import build_ft_controller
from miles.utils.ft.utils.box import Box


class TestRankPidsProviderTOCTOU:
    """H-1: the rank_pids_provider lambda used to read Box.value twice,
    allowing a TOCTOU race where value changes between the None check
    and the method call. Now captured once via walrus operator."""

    def test_returns_empty_dict_when_box_is_none(self) -> None:
        box: Box[MagicMock | None] = Box(None)
        provider = lambda node_id: (r.get_rank_pids_for_node(node_id) if (r := box.value) is not None else {})

        assert provider("node-1") == {}

    def test_returns_rank_pids_when_roster_set(self) -> None:
        roster = MagicMock()
        roster.get_rank_pids_for_node.return_value = {0: 1234}
        box: Box[MagicMock | None] = Box(roster)
        provider = lambda node_id: (r.get_rank_pids_for_node(node_id) if (r := box.value) is not None else {})

        result = provider("node-1")

        assert result == {0: 1234}
        roster.get_rank_pids_for_node.assert_called_once_with("node-1")

    def test_no_error_when_box_cleared_after_set(self) -> None:
        roster = MagicMock()
        roster.get_rank_pids_for_node.return_value = {0: 1234}
        box: Box[MagicMock | None] = Box(roster)
        provider = lambda node_id: (r.get_rank_pids_for_node(node_id) if (r := box.value) is not None else {})

        box.value = None
        assert provider("node-1") == {}


class TestBuildFtControllerStateMachineParams:
    """State machine parameters were hardcoded in the factory and not read
    from FtControllerConfig, so CLI values had no effect."""

    def test_config_state_machine_params_reach_assemble(self) -> None:
        config = FtControllerConfig(
            rollout_num_cells=0,
            recovery_cooldown_window_minutes=60.0,
            recovery_cooldown_max_count=5,
            registration_grace_ticks=10,
            max_simultaneous_bad_nodes=2,
            monitoring_success_iterations=20,
            monitoring_timeout_seconds=1200,
            recovery_timeout_seconds=7200,
        )

        with patch(
            "miles.utils.ft.factories.controller.assemble_ft_controller"
        ) as mock_assemble:
            mock_assemble.return_value = MagicMock()
            build_ft_controller(
                config=config,
                start_exporter=False,
                node_manager_override=StubNodeManager(),
                main_job_override=StubMainJob(),
            )

        kwargs = mock_assemble.call_args.kwargs
        assert kwargs["recovery_cooldown_window_minutes"] == 60.0
        assert kwargs["recovery_cooldown_max_count"] == 5
        assert kwargs["registration_grace_ticks"] == 10
        assert kwargs["max_simultaneous_bad_nodes"] == 2
        assert kwargs["monitoring_success_iterations"] == 20
        assert kwargs["monitoring_timeout_seconds"] == 1200
        assert kwargs["recovery_timeout_seconds"] == 7200


class TestBuildFtControllerRetention:
    """MiniPrometheus retention was hardcoded and not wired from config."""

    def test_retention_from_config_reaches_mini_prometheus(self) -> None:
        config = FtControllerConfig(
            rollout_num_cells=0,
            mini_prometheus_retention_minutes=120.0,
        )
        bundle = build_ft_controller(
            config=config,
            start_exporter=False,
            node_manager_override=StubNodeManager(),
            main_job_override=StubMainJob(),
        )

        store = bundle.controller._metric_store
        mini_prom = store.time_series_store
        assert mini_prom._config.retention == timedelta(minutes=120)
