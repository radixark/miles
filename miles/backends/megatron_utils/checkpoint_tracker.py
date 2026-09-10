from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from miles.backends.megatron_utils.megatron_config import compute_trainer_checkpoint_dir, resolve_megatron_config

CHECKPOINT_TRACKER_FILENAME = "latest_checkpointed_iteration.txt"


def read_trainer_checkpoint_iteration(args: Namespace) -> int | None:
    leader = resolve_megatron_config(args).trainers[0]
    load_dir = (
        compute_trainer_checkpoint_dir(base_dir=args.requested_load, trainer_id=leader.trainer_id)
        if leader.model_id is not None
        else args.requested_load
    )
    return read_checkpoint_tracker_iteration(load_dir)


def read_checkpoint_tracker_iteration(checkpoint_root: str | Path | None) -> int | None:
    if checkpoint_root is None:
        return None

    tracker = Path(checkpoint_root) / CHECKPOINT_TRACKER_FILENAME
    if not tracker.is_file():
        return None

    content = tracker.read_text().strip()
    if not content.isdigit():
        return None
    return int(content)
