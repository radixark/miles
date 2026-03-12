"""Tests for miles.utils.ft.adapters.config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.controller.detectors.chain import DetectorChainConfig


class TestFtControllerConfig:
    def test_defaults(self) -> None:
        config = FtControllerConfig(rollout_num_cells=0)

        assert config.ft_id == ""
        assert config.platform == "stub"
        assert config.tick_interval == 30.0
        assert config.metric_store_backend == "mini"
        assert isinstance(config.detector_config, DetectorChainConfig)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FtControllerConfig(bogus="x")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        config = FtControllerConfig(rollout_num_cells=0)

        with pytest.raises(ValidationError):
            config.ft_id = "changed"  # type: ignore[misc]

    def test_custom_values(self) -> None:
        config = FtControllerConfig(
            rollout_num_cells=0,
            ft_id="my-ft",
            platform="k8s-ray",
            tick_interval=10.0,
            scrape_interval_seconds=5.0,
        )

        assert config.ft_id == "my-ft"
        assert config.platform == "k8s-ray"
        assert config.tick_interval == 10.0
        assert config.scrape_interval_seconds == 5.0

    def test_scrape_interval_seconds_rejects_zero(self) -> None:
        """scrape_interval_seconds was not validated — zero or negative values
        would silently create a broken scrape loop."""
        with pytest.raises(ValidationError, match="must be > 0"):
            FtControllerConfig(rollout_num_cells=0, scrape_interval_seconds=0)

    def test_scrape_interval_seconds_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="must be > 0"):
            FtControllerConfig(rollout_num_cells=0, scrape_interval_seconds=-1.0)

    def test_mini_prometheus_retention_minutes_default(self) -> None:
        config = FtControllerConfig(rollout_num_cells=0)
        assert config.mini_prometheus_retention_minutes == 60.0

    def test_mini_prometheus_retention_minutes_custom(self) -> None:
        config = FtControllerConfig(rollout_num_cells=0, mini_prometheus_retention_minutes=120.0)
        assert config.mini_prometheus_retention_minutes == 120.0

    def test_mini_prometheus_retention_minutes_rejects_zero(self) -> None:
        """Retention was hardcoded in MiniPrometheusConfig and not exposed
        through the config, so it could not be tuned per deployment."""
        with pytest.raises(ValidationError, match="must be > 0"):
            FtControllerConfig(rollout_num_cells=0, mini_prometheus_retention_minutes=0)


class TestStateMachineConfigFields:
    """These parameters were previously only available as factory defaults,
    so operators could not tune them at launch time."""

    def test_state_machine_params_have_sensible_defaults(self) -> None:
        config = FtControllerConfig(rollout_num_cells=0)

        assert config.recovery_cooldown_window_minutes == 30.0
        assert config.recovery_cooldown_max_count == 3
        assert config.registration_grace_ticks == 5
        assert config.max_simultaneous_bad_nodes == 3
        assert config.monitoring_success_iterations == 10
        assert config.monitoring_timeout_seconds == 600
        assert config.recovery_timeout_seconds == 3600
        assert config.rollout_alive_threshold_seconds is None
        assert config.rollout_monitoring_alive_duration_seconds is None

    def test_state_machine_params_accept_custom_values(self) -> None:
        config = FtControllerConfig(
            rollout_num_cells=0,
            recovery_cooldown_window_minutes=60.0,
            recovery_cooldown_max_count=5,
            registration_grace_ticks=10,
            max_simultaneous_bad_nodes=2,
            monitoring_success_iterations=20,
            monitoring_timeout_seconds=1200,
            recovery_timeout_seconds=7200,
            rollout_alive_threshold_seconds=30.0,
            rollout_monitoring_alive_duration_seconds=300.0,
        )

        assert config.recovery_cooldown_window_minutes == 60.0
        assert config.recovery_cooldown_max_count == 5
        assert config.registration_grace_ticks == 10
        assert config.max_simultaneous_bad_nodes == 2
        assert config.monitoring_success_iterations == 20
        assert config.monitoring_timeout_seconds == 1200
        assert config.recovery_timeout_seconds == 7200
        assert config.rollout_alive_threshold_seconds == 30.0
        assert config.rollout_monitoring_alive_duration_seconds == 300.0

    @pytest.mark.parametrize(
        "field_name",
        [
            "recovery_cooldown_max_count",
            "registration_grace_ticks",
            "max_simultaneous_bad_nodes",
            "monitoring_success_iterations",
        ],
    )
    def test_int_params_reject_zero(self, field_name: str) -> None:
        with pytest.raises(ValidationError, match="must be >= 1"):
            FtControllerConfig(rollout_num_cells=0, **{field_name: 0})

    def test_recovery_cooldown_window_rejects_zero(self) -> None:
        with pytest.raises(ValidationError, match="recovery_cooldown_window_minutes"):
            FtControllerConfig(rollout_num_cells=0, recovery_cooldown_window_minutes=0)

    @pytest.mark.parametrize("field_name", ["monitoring_timeout_seconds", "recovery_timeout_seconds"])
    def test_timeout_params_reject_zero(self, field_name: str) -> None:
        with pytest.raises(ValidationError, match="must be > 0"):
            FtControllerConfig(rollout_num_cells=0, **{field_name: 0})


class TestRayJobConfigFields:
    """RayMainJob poll/timeout params were module-level constants, so
    operators could not tune them for slow or high-load clusters."""

    def test_ray_job_params_have_sensible_defaults(self) -> None:
        config = FtControllerConfig(rollout_num_cells=0)

        assert config.ray_job_poll_interval_seconds == 5.0
        assert config.ray_submit_timeout_seconds == 60.0
        assert config.ray_get_status_timeout_seconds == 30.0
        assert config.ray_stop_job_timeout_seconds == 30.0

    def test_ray_job_params_accept_custom_values(self) -> None:
        config = FtControllerConfig(
            rollout_num_cells=0,
            ray_job_poll_interval_seconds=10.0,
            ray_submit_timeout_seconds=120.0,
            ray_get_status_timeout_seconds=60.0,
            ray_stop_job_timeout_seconds=45.0,
        )

        assert config.ray_job_poll_interval_seconds == 10.0
        assert config.ray_submit_timeout_seconds == 120.0
        assert config.ray_get_status_timeout_seconds == 60.0
        assert config.ray_stop_job_timeout_seconds == 45.0

    @pytest.mark.parametrize(
        "field_name",
        [
            "ray_job_poll_interval_seconds",
            "ray_submit_timeout_seconds",
            "ray_get_status_timeout_seconds",
            "ray_stop_job_timeout_seconds",
        ],
    )
    def test_ray_job_params_reject_zero(self, field_name: str) -> None:
        with pytest.raises(ValidationError, match="must be > 0"):
            FtControllerConfig(rollout_num_cells=0, **{field_name: 0})
