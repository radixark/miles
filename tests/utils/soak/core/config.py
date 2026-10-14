from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel

POLL_INTERVAL_SECONDS: float = 2.0
QUIESCENT_POLLS_REQUIRED: int = 60


class SoakTargetConfig(FrozenStrictBaseModel):
    expected_count: int = Field(ge=1)
    mean_interval_seconds: float = Field(gt=0, allow_inf_nan=False)


class SoakTimeoutConfig(FrozenStrictBaseModel):
    run_seconds: float = Field(default=21600.0, gt=0, allow_inf_nan=False)
    tail_seconds: float = Field(default=3600.0, gt=0, allow_inf_nan=False)
    observation_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    final_observation_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)


class SoakTailConfig(FrozenStrictBaseModel):
    close_after_rollout_id: int = Field(ge=0)

    @classmethod
    def create(cls, *, num_rollout: int, min_tail_rollouts: int = 3) -> "SoakTailConfig":
        raise NotImplementedError


class SoakRunnerConfig(FrozenStrictBaseModel):
    seed: int
    start_after_rollout_id: int | None = Field(default=None, ge=0)
    target_configs: dict[str, SoakTargetConfig] = Field(default_factory=dict)
    timeouts: SoakTimeoutConfig = Field(default_factory=SoakTimeoutConfig)
    tail: SoakTailConfig | None = None
    poll_interval_seconds: float = Field(default=POLL_INTERVAL_SECONDS, gt=0, allow_inf_nan=False)
    quiescent_polls_required: int = Field(default=QUIESCENT_POLLS_REQUIRED, ge=1)
