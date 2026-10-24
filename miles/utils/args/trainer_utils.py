from argparse import Namespace

from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.megatron_utils.megatron_config import (
    MegatronArgsNamespace,
    MegatronTrainerConfig,
    compute_trainer_args,
)
from miles.backends.torchtitan_utils.args_namespace import TorchtitanArgsNamespace
from miles.utils.args.configs.backend_fields import TrainerBackendTraitConfig
from miles.utils.args.runtime import AllConfig, TrainerConfig


# TODO: After zhichen's training backend refactor, make per-trainer argument computation consume structured configs without flattening Miles and backend fields.
def compute_trainer_config(all_config: AllConfig, trainer: MegatronTrainerConfig) -> TrainerConfig:
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

    values["backend"] = backend_cls(
        **{name: values[name] for name in base_backend_values.keys() | TrainerBackendTraitConfig.model_fields.keys()}
    )

    return TrainerConfig.model_validate({name: values[name] for name in TrainerConfig.model_fields if name in values})
