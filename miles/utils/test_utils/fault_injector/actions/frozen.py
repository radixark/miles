import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Literal

from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.fault_injector.actions.base import BaseFaultAction, FaultHookContext, FaultHookResources

logger = logging.getLogger(__name__)

SLEEP_FOREVER_INTERVAL_SECONDS: float = 60.0
PARKABLE_TRAIN_SCRIPT: str = "train.py"
FROZEN_SENTINEL_SUFFIX: str = "_frozen_at.json"


class SleepForeverAction(BaseFaultAction):
    kind: Literal["sleep_forever"] = "sleep_forever"

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        _assert_parkable(resources.args, trainer_model_id=context.trainer_model_id)
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


def _assert_parkable(args: object, *, trainer_model_id: str | None) -> None:
    assert (script := Path(sys.argv[0]).name) == PARKABLE_TRAIN_SCRIPT, (
        f"sleep_forever parks the orchestration script between two steps, and only {PARKABLE_TRAIN_SCRIPT} stands "
        f"still at that point; {script} has already started the next rollout by the time it reaches here, so the run "
        f"would not be standing where the hook names"
    )
    assert (interval := args.update_weights_interval) == 1, (
        f"sleep_forever parks the run where it updates weights, and --update-weights-interval {interval} means the "
        f"run does not pass through that point after every step"
    )
    assert trainer_model_id is None, (
        f"sleep_forever parks one coroutine, and a run training several policies drives one per policy "
        f"({trainer_model_id!r} reached it here), so every other policy would keep training past the step the hook "
        f"names"
    )
