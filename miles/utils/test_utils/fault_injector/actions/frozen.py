import asyncio
import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import Field

from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.fault_injector.actions.base import (
    BaseFaultAction,
    FaultHookContext,
    FaultHookResources,
    _assert_synchronous_orchestrator,
)

logger = logging.getLogger(__name__)

SLEEP_FOREVER_INTERVAL_SECONDS: float = 60.0
FROZEN_SENTINEL_SUFFIX: str = "_frozen_at.json"


class SleepAction(BaseFaultAction):
    kind: Literal["sleep"] = "sleep"
    seconds: float = Field(gt=0)

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        logger.warning(f"Fault hook: sleeping {self.seconds}s at rollout {context.rollout_id}")
        await asyncio.sleep(self.seconds)


class SleepForeverAction(BaseFaultAction):
    kind: Literal["sleep_forever"] = "sleep_forever"

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        _assert_synchronous_orchestrator(resources.args, trainer_model_id=context.trainer_model_id, action=self.kind)
        msg = f"Fault hook: sleeping forever after rollout {context.rollout_id}; the run never starts the next step"
        logger.warning(msg)
        print(msg, flush=True)
        if (hooks_path := resources.args.ci_fault_hooks_path) is not None:
            assert context.rollout_id is not None
            write_frozen_sentinel(Path(hooks_path), rollout_id=context.rollout_id)
        while True:
            await asyncio.sleep(SLEEP_FOREVER_INTERVAL_SECONDS)


def compute_frozen_sentinel_path(hooks_path: Path) -> Path:
    return hooks_path.with_name(f"{hooks_path.stem}{FROZEN_SENTINEL_SUFFIX}")


def write_frozen_sentinel(hooks_path: Path, *, rollout_id: int) -> None:
    path = compute_frozen_sentinel_path(hooks_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps({"rollout_id": rollout_id}))


def read_frozen_rollout_id(hooks_path: Path) -> int | None:
    path = compute_frozen_sentinel_path(hooks_path)
    if not path.is_file():
        return None
    return int(json.loads(path.read_text())["rollout_id"])
