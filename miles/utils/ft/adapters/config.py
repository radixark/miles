from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field, field_validator

from miles.utils.ft.controller.detectors.chain import DetectorChainConfig
from miles.utils.ft.utils.base_model import FtBaseModel

if TYPE_CHECKING:
    from miles.utils.ft.controller.runtime_config import ControllerRuntimeConfig


class FtControllerConfig(FtBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ft_id: str = ""
    k8s_label_prefix: str = ""
    k8s_namespace: str = ""
    platform: Literal["stub", "k8s-ray"] = "stub"
    ray_address: str = "http://127.0.0.1:8265"
    entrypoint: str = ""
    runtime_env: dict[str, Any] = Field(default_factory=dict)
    metric_store_backend: Literal["mini", "prometheus"] = "mini"
    prometheus_url: str = "http://prometheus:9090"
    controller_exporter_port: int = 0
    tick_interval: float = 10.0
    scrape_interval_seconds: float = 10.0
    mini_prometheus_retention_minutes: float = 60.0
    notify_webhook_url: str = ""
    notify_platform: str = ""
    notify_timeout_seconds: float = 10.0
    notify_max_retries: int = 3
    notify_initial_backoff_seconds: float = 1.0
    rollout_num_cells: int
    detector_config: DetectorChainConfig = Field(default_factory=DetectorChainConfig)

    recovery_cooldown_window_minutes: float = 30.0
    recovery_cooldown_max_count: int = 3
    registration_grace_ticks: int = 5
    max_simultaneous_bad_nodes: int = 3
    monitoring_success_iterations: int = 10
    monitoring_timeout_seconds: int = 600
    recovery_timeout_seconds: int = 3600
    rollout_alive_threshold_seconds: float | None = None
    rollout_monitoring_alive_duration_seconds: float | None = None

    ray_job_poll_interval_seconds: float = 5.0
    ray_submit_timeout_seconds: float = 60.0
    ray_get_status_timeout_seconds: float = 30.0
    ray_stop_job_timeout_seconds: float = 30.0

    @field_validator(
        "scrape_interval_seconds",
        "mini_prometheus_retention_minutes",
        "ray_job_poll_interval_seconds",
        "ray_submit_timeout_seconds",
        "ray_get_status_timeout_seconds",
        "ray_stop_job_timeout_seconds",
        "notify_timeout_seconds",
        "notify_initial_backoff_seconds",
    )
    @classmethod
    def _positive_float_fields(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @field_validator("recovery_cooldown_window_minutes")
    @classmethod
    def _cooldown_window_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("recovery_cooldown_window_minutes must be > 0")
        return v

    @field_validator("recovery_cooldown_max_count", "max_simultaneous_bad_nodes", "notify_max_retries")
    @classmethod
    def _int_params_must_be_at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("registration_grace_ticks", "monitoring_success_iterations")
    @classmethod
    def _int_params_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    @field_validator("monitoring_timeout_seconds", "recovery_timeout_seconds")
    @classmethod
    def _timeout_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    def to_runtime_config(self) -> ControllerRuntimeConfig:
        from miles.utils.ft.controller.runtime_config import ControllerRuntimeConfig

        return ControllerRuntimeConfig(
            tick_interval=self.tick_interval,
            recovery_cooldown_window_minutes=self.recovery_cooldown_window_minutes,
            recovery_cooldown_max_count=self.recovery_cooldown_max_count,
            registration_grace_ticks=self.registration_grace_ticks,
            max_simultaneous_bad_nodes=self.max_simultaneous_bad_nodes,
            monitoring_success_iterations=self.monitoring_success_iterations,
            monitoring_timeout_seconds=self.monitoring_timeout_seconds,
            recovery_timeout_seconds=self.recovery_timeout_seconds,
            rollout_alive_threshold_seconds=self.rollout_alive_threshold_seconds,
            rollout_monitoring_alive_duration_seconds=self.rollout_monitoring_alive_duration_seconds,
        )
