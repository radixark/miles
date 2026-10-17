from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.schema import BaseConfig


class TrainerOnlyConfig(BaseConfig):
    backend: MegatronArgsNamespace
    trainer_role: str
