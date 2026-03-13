"""Tests for controller/factory.py — rank_pids_provider lambda safety and config wiring."""

from __future__ import annotations

import os
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.adapters.stubs import StubMainJob, StubNodeManager
from miles.utils.ft.controller.runtime_config import ControllerRuntimeConfig
from miles.utils.ft.factories.controller.backends import build_platform_components
from miles.utils.ft.factories.controller.from_config import build_ft_controller
from miles.utils.ft.utils.box import Box


class TestRankPidsProviderTOCTOU:
    """H-1: the rank_pids_provider lambda used to read Box.value twice,
    allowing a TOCTOU race where value changes between the None check
    and the method call. Now captured once via walrus operator."""

    def test_returns_empty_dict_when_box_is_none(self) -> None:
        box: Box[MagicMock | None] = Box(None)

        def provider(node_id: str) -> dict[int, int]:
            roster = box.value
            return roster.get_rank_pids_for_node(node_id) if roster is not None else {}

        assert provider("node-1") == {}

    def test_returns_rank_pids_when_roster_set(self) -> None:
        roster = MagicMock()
        roster.get_rank_pids_for_node.return_value = {0: 1234}
        box: Box[MagicMock | None] = Box(roster)

        def provider(node_id: str) -> dict[int, int]:
            current_roster = box.value
            return current_roster.get_rank_pids_for_node(node_id) if current_roster is not None else {}

        result = provider("node-1")

        assert result == {0: 1234}
        roster.get_rank_pids_for_node.assert_called_once_with("node-1")

    def test_no_error_when_box_cleared_after_set(self) -> None:
        roster = MagicMock()
        roster.get_rank_pids_for_node.return_value = {0: 1234}
        box: Box[MagicMock | None] = Box(roster)

        def provider(node_id: str) -> dict[int, int]:
            current_roster = box.value
            return current_roster.get_rank_pids_for_node(node_id) if current_roster is not None else {}

        box.value = None
        assert provider("node-1") == {}


class TestBuildFtControllerStateMachineParams:
    """ControllerRuntimeConfig is derived from FtControllerConfig and passed through to assemble."""

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

        with patch("miles.utils.ft.factories.controller.from_config.assemble_ft_controller") as mock_assemble:
            mock_assemble.return_value = MagicMock()
            build_ft_controller(
                config=config,
                start_exporter=False,
                node_manager_override=StubNodeManager(),
                main_job_override=StubMainJob(),
            )

        args = mock_assemble.call_args
        runtime_config = args[0][0]
        assert isinstance(runtime_config, ControllerRuntimeConfig)
        assert runtime_config.recovery_cooldown_window_minutes == 60.0
        assert runtime_config.recovery_cooldown_max_count == 5
        assert runtime_config.registration_grace_ticks == 10
        assert runtime_config.max_simultaneous_bad_nodes == 2
        assert runtime_config.monitoring_success_iterations == 20
        assert runtime_config.monitoring_timeout_seconds == 1200
        assert runtime_config.recovery_timeout_seconds == 7200


class TestRolloutMonitoringTimeoutWiring:
    """Rollout subsystem previously did not receive monitoring_timeout_seconds
    from the global config. Its MonitoringRunningAfterDelayConfig always used
    the default 600s regardless of what the user configured."""

    def test_rollout_monitoring_uses_global_timeout(self) -> None:
        config = FtControllerConfig(
            rollout_num_cells=2,
            monitoring_timeout_seconds=1800,
        )
        bundle = build_ft_controller(
            config=config,
            start_exporter=False,
            node_manager_override=StubNodeManager(),
            main_job_override=StubMainJob(),
        )

        from miles.utils.ft.controller.state_machines.restart.models import MonitoringRunningAfterDelayConfig

        for name, spec in bundle.controller._subsystem_specs.items():
            if name.startswith("rollout_"):
                assert isinstance(spec.config.monitoring_config, MonitoringRunningAfterDelayConfig)
                assert spec.config.monitoring_config.timeout_seconds == 1800

    def test_training_and_rollout_share_same_timeout(self) -> None:
        config = FtControllerConfig(
            rollout_num_cells=1,
            monitoring_timeout_seconds=900,
        )
        bundle = build_ft_controller(
            config=config,
            start_exporter=False,
            node_manager_override=StubNodeManager(),
            main_job_override=StubMainJob(),
        )

        specs = bundle.controller._subsystem_specs
        training_timeout = specs["training"].config.monitoring_config.timeout_seconds
        rollout_timeout = specs["rollout_0"].config.monitoring_config.timeout_seconds
        assert training_timeout == rollout_timeout == 900


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


class TestK8sNamespaceFallback:
    """K8s namespace was only read from K8S_NAMESPACE env var. Now the CLI
    parameter takes precedence, with the env var as fallback."""

    def test_cli_namespace_takes_precedence_over_env_var(self) -> None:
        with patch.dict(os.environ, {"K8S_NAMESPACE": "env-ns"}):
            node_mgr, _ = build_platform_components(
                platform="k8s-ray",
                ray_address="http://localhost:8265",
                entrypoint="python train.py",
                ft_id="test",
                k8s_label_prefix="",
                k8s_namespace="cli-ns",
            )
        assert node_mgr._namespace == "cli-ns"

    def test_falls_back_to_env_var_when_cli_empty(self) -> None:
        with patch.dict(os.environ, {"K8S_NAMESPACE": "env-ns"}):
            node_mgr, _ = build_platform_components(
                platform="k8s-ray",
                ray_address="http://localhost:8265",
                entrypoint="python train.py",
                ft_id="test",
                k8s_label_prefix="",
                k8s_namespace="",
            )
        assert node_mgr._namespace == "env-ns"

    def test_raises_when_both_cli_and_env_var_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="namespace not configured"):
                build_platform_components(
                    platform="k8s-ray",
                    ray_address="http://localhost:8265",
                    entrypoint="python train.py",
                    ft_id="test",
                    k8s_label_prefix="",
                    k8s_namespace="",
                )
