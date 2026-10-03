from argparse import Namespace

from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.megatron_utils.checkpoint_request import CHECKPOINT_LOAD_FIELDS, MegatronCheckpointLoad
from miles.backends.megatron_utils.megatron_config import (
    CRITIC_ROLE,
    MegatronArgsNamespace,
    MegatronTrainerConfig,
    compute_trainer_args,
    compute_trainer_checkpoint_dir,
)
from miles.utils.args.configs.backend_fields import TrainerBackendTraitConfig
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.runtime import AllConfig, OrchestratorConfig, TrainerConfig


# TODO: After zhichen's training backend refactor, make per-trainer argument computation consume structured configs without flattening Miles and backend fields.
def compute_trainer_config(
    all_config: AllConfig | OrchestratorConfig, trainer: MegatronTrainerConfig
) -> TrainerConfig:
    base_backend_values = (
        all_config.raw_megatron.base_args if all_config.train_backend == "megatron" else vars(all_config.raw_fsdp)
    )
    base_args = Namespace(**(dict(all_config) | base_backend_values))
    values = vars(compute_trainer_args(args=base_args, trainer=trainer))
    backend_names = base_backend_values.keys() | TrainerBackendTraitConfig.model_fields.keys()
    if all_config.train_backend == "megatron":
        backend_names -= CHECKPOINT_LOAD_FIELDS
    backend_cls = MegatronArgsNamespace if all_config.train_backend == "megatron" else FsdpArgsNamespace
    values["backend"] = backend_cls(**{name: values[name] for name in backend_names})
    return TrainerConfig.model_validate({name: values[name] for name in TrainerConfig.model_fields if name in values})


def compute_trainer_checkpoint_load(
    all_config: AllConfig | OrchestratorConfig, trainer: MegatronTrainerConfig
) -> MegatronCheckpointLoad | None:
    if all_config.train_backend != "megatron":
        return None
    values = dict(all_config) | all_config.raw_megatron.base_args | trainer.overrides
    if trainer.model_id is not None:
        values["load"] = compute_trainer_checkpoint_dir(base_dir=values["load"], trainer_id=trainer.trainer_id)
    return MegatronCheckpointLoad.from_args(Namespace(**values))


# TODO: support different sizes after the args refactor
def compute_trainer_total_gpus(scaling: ScalingConfig, *, role: str) -> int:
    if role == CRITIC_ROLE:
        return scaling.critic_num_nodes * scaling.critic_num_gpus_per_node
    return scaling.actor_num_nodes * scaling.actor_num_gpus_per_node
