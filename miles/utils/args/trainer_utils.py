import copy
from argparse import Namespace

from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.megatron_utils.checkpoint_request import CHECKPOINT_LOAD_FIELDS, MegatronCkptLoadConfig
from miles.backends.megatron_utils.megatron_config import (
    CRITIC_ROLE,
    MegatronArgsNamespace,
    MegatronTrainerConfig,
    compute_trainer_args,
    compute_trainer_checkpoint_dir,
    resolve_args_checkpoint_load,
)
from miles.backends.torchtitan_utils.args_namespace import TorchtitanArgsNamespace
from miles.utils.args.configs.backend_fields import TrainerBackendTraitConfig
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.runtime import AllConfig, OrchestratorConfig, TrainerConfig


# TODO: After zhichen's training backend refactor, make per-trainer argument computation consume structured configs without flattening Miles and backend fields.
def compute_trainer_config(
    all_config: AllConfig | OrchestratorConfig, trainer: MegatronTrainerConfig
) -> TrainerConfig:
    if all_config.train_backend == "megatron":
        base_backend_values = all_config.raw_megatron.base_args
        backend_cls = MegatronArgsNamespace
    elif all_config.train_backend == "fsdp":
        assert all_config.raw_fsdp is not None
        base_backend_values = vars(all_config.raw_fsdp)
        backend_cls = FsdpArgsNamespace
    else:
        assert all_config.train_backend == "torchtitan"
        assert all_config.raw_torchtitan is not None
        base_backend_values = vars(all_config.raw_torchtitan)
        backend_cls = TorchtitanArgsNamespace
    base_args = Namespace(**(dict(all_config) | base_backend_values))
    values = vars(compute_trainer_args(args=base_args, trainer=trainer))
    backend_names = base_backend_values.keys() | TrainerBackendTraitConfig.model_fields.keys()
    if all_config.train_backend == "megatron":
        backend_names -= CHECKPOINT_LOAD_FIELDS
    values["backend"] = backend_cls(**{name: values[name] for name in backend_names})
    return TrainerConfig.model_validate({name: values[name] for name in TrainerConfig.model_fields if name in values})


def compute_trainer_ckpt_load_config(
    all_config: AllConfig | OrchestratorConfig, trainer: MegatronTrainerConfig
) -> MegatronCkptLoadConfig | None:
    if all_config.train_backend != "megatron":
        return None
    args = copy.deepcopy(Namespace(**(dict(all_config) | all_config.raw_megatron.base_args | trainer.overrides)))
    if trainer.model_id is not None:
        args.load = compute_trainer_checkpoint_dir(base_dir=args.load, trainer_id=trainer.trainer_id)

    resume_from_ckpt = resolve_args_checkpoint_load(args, role=trainer.role)
    return MegatronCkptLoadConfig(
        load=args.load,
        resume_from_ckpt=resume_from_ckpt,
        no_load_optim=args.no_load_optim,
        no_load_rng=args.no_load_rng,
        finetune=args.finetune,
        ckpt_step=args.ckpt_step,
    )


# TODO: support different sizes after the args refactor
def compute_trainer_total_gpus(scaling: ScalingConfig, *, role: str) -> int:
    if role == CRITIC_ROLE:
        return scaling.critic_num_nodes * scaling.critic_num_gpus_per_node
    return scaling.actor_num_nodes * scaling.actor_num_gpus_per_node
