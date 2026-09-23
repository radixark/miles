import abc
from dataclasses import dataclass

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class FaultHookContext(FrozenStrictBaseModel):
    weight_version: int | None = None
    rollout_id: int | None = None
    attempt: int | None = None
    trainer_model_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class FaultHookResources:
    args: object | None = None


class BaseFaultAction(FrozenStrictBaseModel, abc.ABC):
    @abc.abstractmethod
    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None: ...
