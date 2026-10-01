from typing import Self

from miles.utils.args.runtime import AllConfig, OrchestratorConfig
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class TrainerControllerInitRequest(FrozenStrictBaseModel):
    num_rollout: int | None
    wandb_run_id: str | None
    mlflow_run_id: str | None

    @classmethod
    def from_args(cls, args: OrchestratorConfig | AllConfig) -> Self:
        return cls(num_rollout=args.num_rollout, wandb_run_id=args.wandb_run_id, mlflow_run_id=args.mlflow_run_id)
