import abc
import asyncio
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel

if TYPE_CHECKING:
    from miles.ray.train.group import TrainerController
    from miles.utils.workers.cell_operations.base import BaseCellOperations


class FaultHookContext(FrozenStrictBaseModel):
    weight_version: int | None = None
    debug_weight_update_id: str | None = None
    snapshot_cell_id_to_hashes: dict[str, str] = Field(default_factory=dict)
    rollout_id: int | None = None
    attempt: int | None = None
    trainer_model_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class FaultHookResources:
    args: object | None = None
    controller: "TrainerController | None" = None
    cell_operations: "BaseCellOperations | None" = None
    managed_process: subprocess.Popen | None = None
    owner_loop: asyncio.AbstractEventLoop | None = None


class BaseFaultAction(FrozenStrictBaseModel, abc.ABC):
    @abc.abstractmethod
    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None: ...


PARKABLE_TRAIN_SCRIPT: str = "train.py"


def _assert_synchronous_orchestrator(args: object, trainer_model_id: str | None, action: str) -> None:
    assert (script := Path(sys.argv[0]).name) == PARKABLE_TRAIN_SCRIPT, (
        f"{action} runs between synchronous steps, and only {PARKABLE_TRAIN_SCRIPT} stands still there; "
        f"{script} has already started the next rollout by the time it reaches here"
    )
    assert (interval := args.update_weights_interval) == 1, (
        f"{action} runs where weights update, and --update-weights-interval {interval} means the run does not "
        "pass through that point after every step"
    )
    assert trainer_model_id is None, (
        f"{action} pauses one coroutine, and a run training several policies drives one per policy "
        f"({trainer_model_id!r} reached it here), so every other policy would keep training past this step"
    )
