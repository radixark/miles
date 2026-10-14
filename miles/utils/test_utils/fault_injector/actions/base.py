import abc
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from miles.utils.pydantic_utils import FrozenStrictBaseModel

if TYPE_CHECKING:
    from miles.ray.train.group import TrainerController
    from miles.utils.workers.cell_operations.base import BaseCellOperations


class FaultHookContext(FrozenStrictBaseModel):
    weight_version: int | None = None
    rollout_id: int | None = None
    attempt: int | None = None
    trainer_model_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class FaultHookResources:
    args: object | None = None
    controller: "TrainerController | None" = None
    cell_operations: "BaseCellOperations | None" = None
    managed_process: subprocess.Popen | None = None


class BaseFaultAction(FrozenStrictBaseModel, abc.ABC):
    @abc.abstractmethod
    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None: ...
