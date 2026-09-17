from argparse import Namespace

from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.megatron_utils.megatron_config import (
    MegatronArgsNamespace,
    MegatronTrainerConfig,
    compute_trainer_args,
)
from miles.utils.args.configs.backend_fields import TrainerBackendTraitConfig
from miles.utils.args.runtime import AllConfig, TrainerConfig


# TODO: After zhichen's training backend refactor, make per-trainer argument computation consume structured configs without flattening Miles and backend fields.
def compute_trainer_config(all_config: AllConfig, trainer: MegatronTrainerConfig) -> TrainerConfig:
    base_backend_values = (
        all_config.raw_megatron.base_args if all_config.train_backend == "megatron" else vars(all_config.raw_fsdp)
    )
    base_args = Namespace(**(dict(all_config) | base_backend_values))
    values = vars(compute_trainer_args(args=base_args, trainer=trainer))

    backend_cls = MegatronArgsNamespace if all_config.train_backend == "megatron" else FsdpArgsNamespace
    values["backend"] = backend_cls(
        **{name: values[name] for name in base_backend_values.keys() | TrainerBackendTraitConfig.model_fields.keys()}
    )

    return TrainerConfig.model_validate({name: values[name] for name in TrainerConfig.model_fields if name in values})
