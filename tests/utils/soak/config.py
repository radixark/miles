from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class SoakCellPolicy(FrozenStrictBaseModel):
    expected_cells: int | None = Field(default=None, ge=2)


class SoakPolicy(FrozenStrictBaseModel):
    start_after_rollout_id: int | None = Field(default=None, ge=0)
    cell_policies: dict[str, SoakCellPolicy] = Field(default_factory=dict)


class SoakTimeouts(FrozenStrictBaseModel):
    run_seconds: float = Field(default=21600.0, gt=0, allow_inf_nan=False)
    tail_seconds: float = Field(default=3600.0, gt=0, allow_inf_nan=False)
    observation_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    final_observation_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)


class SoakTailPolicy(FrozenStrictBaseModel):
    close_after_rollout_id: int = Field(ge=0)
    trainer_id: str = "actor"


def create_tail_policy(*, num_rollout: int, min_tail_rollouts: int = 3) -> SoakTailPolicy:
    if min_tail_rollouts < 3 or num_rollout <= min_tail_rollouts:
        raise ValueError("A soak needs an injection rollout followed by its complete recovery tail")
    return SoakTailPolicy(close_after_rollout_id=num_rollout - max(min_tail_rollouts, num_rollout // 5) - 1)


def create_policy(
    *,
    expected_cells: dict[str, int],
) -> SoakPolicy:
    return SoakPolicy(
        cell_policies={kind: SoakCellPolicy(expected_cells=count) for kind, count in expected_cells.items()},
    )
