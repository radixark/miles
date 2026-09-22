from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class SoakTargetPolicy(FrozenStrictBaseModel):
    expected_count: int = Field(ge=1)


class SoakTimeouts(FrozenStrictBaseModel):
    run_seconds: float = Field(default=21600.0, gt=0, allow_inf_nan=False)
    tail_seconds: float = Field(default=3600.0, gt=0, allow_inf_nan=False)
    observation_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    final_observation_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)


class SoakTailPolicy(FrozenStrictBaseModel):
    close_after_rollout_id: int = Field(ge=0)


class SoakRunnerConfig(FrozenStrictBaseModel):
    start_after_rollout_id: int | None = Field(default=None, ge=0)
    target_policies: dict[str, SoakTargetPolicy] = Field(default_factory=dict)
    timeouts: SoakTimeouts = Field(default_factory=SoakTimeouts)
    tail: SoakTailPolicy | None = None


def create_tail_policy(*, num_rollout: int, min_tail_rollouts: int = 3) -> SoakTailPolicy:
    if min_tail_rollouts < 3 or num_rollout <= min_tail_rollouts:
        raise ValueError("A soak needs an injection rollout followed by its complete recovery tail")
    return SoakTailPolicy(close_after_rollout_id=num_rollout - max(min_tail_rollouts, num_rollout // 5) - 1)
