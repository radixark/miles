from typing import Self

from miles.backends.megatron_utils.checkpoint_request import MegatronCheckpointLoad
from miles.backends.megatron_utils.megatron_config import MegatronTrainerConfig
from miles.utils.args.runtime import AllConfig, OrchestratorConfig
from miles.utils.args.trainer_utils import compute_trainer_checkpoint_load
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class TrainerControllerInitRequest(FrozenStrictBaseModel):
    num_rollout: int | None
    wandb_run_id: str | None
    mlflow_run_id: str | None
    checkpoint_load: MegatronCheckpointLoad | None = None

    @classmethod
    def from_args(cls, args: OrchestratorConfig | AllConfig, *, trainer: MegatronTrainerConfig) -> Self:
        return cls(
            num_rollout=args.num_rollout,
            wandb_run_id=args.wandb_run_id,
            mlflow_run_id=args.mlflow_run_id,
            checkpoint_load=compute_trainer_checkpoint_load(args, trainer),
        )
