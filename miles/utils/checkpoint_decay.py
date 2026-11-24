"""Delete old checkpoint steps, keeping the newest few and every Nth older one."""

import logging
import pathlib
import re
import shutil

from miles.utils import misc

logger = logging.getLogger(__name__)

_ITER_DIR = re.compile(r"iter_(\d{7})")


def _pin_marker(root: pathlib.Path, step: int) -> pathlib.Path:
    return root / f"iter_{step:07d}.keep"


def prune_checkpoints(
    save_dir: str, *, saved_step: int, keep_recent: int, keep_every: int, pin_saved: bool = False
) -> None:
    """Remove steps up to `saved_step` unless among the newest `keep_recent` or ending every `keep_every`th rollout.

    Step N is rollout N; it is kept on the same cadence `--save-interval` saves on, so every kept step was saved.

    A step is megatron's `iter_NNNNNNN` directory plus the rollout side's `rollout/N`; both go together. Steps
    above `saved_step` (left by a run resumed from an earlier `--ckpt-step`) are not touched.

    A step that cannot be removed (a symlink, an NFS busy file) is logged and kept, so housekeeping never ends a
    run whose new checkpoint is already written. `iter_NNNNNNN` goes first, so a failure keeps `rollout/N`.

    A step with a sibling `iter_NNNNNNN.keep` marker is never removed. `pin_saved` writes that marker for
    `saved_step` first (used for sentinel-triggered saves); a marker can also be created or deleted by hand.
    """
    root = pathlib.Path(save_dir)
    if pin_saved:
        _pin_marker(root, saved_step).touch()
    steps = sorted(
        step for p in root.iterdir() if (m := _ITER_DIR.fullmatch(p.name)) and (step := int(m.group(1))) <= saved_step
    )
    for step in steps[:-keep_recent]:
        if misc.should_run_periodic_action(step, keep_every) or _pin_marker(root, step).exists():
            continue
        try:
            for directory in (root / f"iter_{step:07d}", root / "rollout" / str(step)):
                if directory.is_dir():
                    logger.info(f"checkpoint decay: removing {directory}")
                    shutil.rmtree(directory)
        except OSError:
            logger.exception(f"checkpoint decay: could not remove step {step}; keeping it")
