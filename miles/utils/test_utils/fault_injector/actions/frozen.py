import json
import os
import sys
from pathlib import Path

SLEEP_FOREVER_AT_END_ACTION: str = "sleep_forever_at_end"
PARKABLE_TRAIN_SCRIPT: str = "train.py"
FROZEN_SENTINEL_SUFFIX: str = "_frozen_at.json"


# TODO ad hoc hack: revert after the args refactor
def compute_frozen_sentinel_path(actions_path: Path) -> Path:
    return actions_path.with_name(f"{actions_path.stem}{FROZEN_SENTINEL_SUFFIX}")


# TODO ad hoc hack: revert after the args refactor
def write_frozen_sentinel(actions_path: Path, *, rollout_id: int) -> None:
    path = compute_frozen_sentinel_path(actions_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scratch = path.with_name(f"{path.name}.{os.getpid()}.partial")
    scratch.write_text(json.dumps({"rollout_id": rollout_id}))
    scratch.replace(path)


# TODO ad hoc hack: revert after the args refactor
def read_frozen_rollout_id(actions_path: Path) -> int | None:
    path = compute_frozen_sentinel_path(actions_path)
    if not path.is_file():
        return None
    return int(json.loads(path.read_text())["rollout_id"])


def assert_loop_parkable(args: object, *, trainer_model_id: str | None) -> None:
    assert (script := Path(sys.argv[0]).name) == PARKABLE_TRAIN_SCRIPT, (
        f"{SLEEP_FOREVER_AT_END_ACTION} parks the orchestration script between two steps, and only "
        f"{PARKABLE_TRAIN_SCRIPT} stands still at that point; {script} has already started the next rollout by the "
        f"time it reaches here, so the run would not be standing where the action names"
    )
    assert (interval := args.update_weights_interval) == 1, (
        f"{SLEEP_FOREVER_AT_END_ACTION} parks the run where it updates weights, and --update-weights-interval "
        f"{interval} means the run does not pass through that point after every step"
    )
    assert trainer_model_id is None, (
        f"{SLEEP_FOREVER_AT_END_ACTION} parks one coroutine, and a run training several policies drives one per "
        f"policy ({trainer_model_id!r} reached it here), so every other policy would keep training past the step "
        f"the action names"
    )
