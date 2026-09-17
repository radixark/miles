from typing import Annotated, Any

from pydantic import Discriminator, Tag

from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.schema import BaseConfig


def _backend_name(value: Any) -> str:
    if isinstance(value, dict):
        return value["backend_name"]
    if isinstance(value, (MegatronArgsNamespace, FsdpArgsNamespace)):
        return value.backend_name
    raise TypeError(f"Unsupported backend configuration type: {type(value).__name__}")


class TrainerOnlyConfig(BaseConfig):
    backend: Annotated[
        Annotated[MegatronArgsNamespace, Tag("megatron")] | Annotated[FsdpArgsNamespace, Tag("fsdp")],
        Discriminator(_backend_name),
    ]
    trainer_role: str
