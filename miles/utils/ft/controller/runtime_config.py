from __future__ import annotations

from pydantic import ConfigDict, field_validator

from miles.utils.ft.utils.base_model import FtBaseModel


class ControllerRuntimeConfig(FtBaseModel):
    """Assembly-time and runtime scalar configuration for FtController.

    This is the subset of FtControllerConfig that assemble_ft_controller
    and FtController actually consume.  Tests can construct this directly
    without specifying platform/k8s/notifier fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tick_interval: float = 10.0
    recovery_cooldown_window_minutes: float = 30.0
    recovery_cooldown_max_count: int = 3
    registration_grace_ticks: int = 5
    max_simultaneous_bad_nodes: int = 3
    monitoring_success_iterations: int = 10
    monitoring_timeout_seconds: int = 600
    recovery_timeout_seconds: int = 3600
    rollout_alive_threshold_seconds: float | None = None
    rollout_monitoring_alive_duration_seconds: float | None = None

    @field_validator("recovery_cooldown_window_minutes")
    @classmethod
    def _cooldown_window_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @field_validator("registration_grace_ticks", "monitoring_success_iterations")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    @field_validator("recovery_cooldown_max_count", "max_simultaneous_bad_nodes")
    @classmethod
    def _at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("monitoring_timeout_seconds", "recovery_timeout_seconds")
    @classmethod
    def _timeout_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v
