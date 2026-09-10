"""Snapshot/restore the event directory alongside model checkpoints."""

import logging
import shutil
import time
import uuid
from argparse import Namespace
from pathlib import Path

from miles.backends.megatron_utils.checkpoint_tracker import read_trainer_checkpoint_iteration

logger = logging.getLogger(__name__)


def snapshot(args: Namespace, iteration: int) -> None:
    if args.save_debug_event_data is None or args.save is None:
        return

    src = Path(args.save_debug_event_data)
    if not src.is_dir():
        return

    dst = _snapshot_dir(Path(args.save), iteration)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    logger.info("Snapshotted event dir %s -> %s", src, dst)


def restore(args: Namespace) -> None:
    if args.save_debug_event_data is None:
        return

    src = None
    if args.requested_load is not None and (iteration := _read_checkpoint_iteration(args)) is not None:
        src = _snapshot_dir(checkpoint_root=Path(args.requested_load), iteration=iteration)

    dst = Path(args.save_debug_event_data)
    if src is None or not src.is_dir():
        if dst.is_dir() and any(dst.iterdir()):
            trash = _move_aside(dst)
            dst.mkdir(parents=True)
            logger.info("Moved the log of the run a fresh start replaces %s -> %s", dst, trash)
        return

    if dst.exists():
        trash = _move_aside(dst)
        logger.info("Moved pre-restore event dir %s -> %s", dst, trash)
    shutil.copytree(src, dst)
    logger.info("Restored event dir %s <- %s", dst, src)


def discard(args: Namespace) -> None:
    if args.save_debug_event_data is None:
        return

    dst = Path(args.save_debug_event_data)
    assert not dst.is_symlink(), f"a take-over moves the event log aside, and {dst} is a symlink"
    if not dst.is_dir() or not any(dst.iterdir()):
        return

    # TODO: startup events the new incarnation already wrote go into the trash with the abandoned log
    trash = _move_aside(dst)
    dst.mkdir(parents=True)
    logger.info("Moved the log of the run a hot restart takes over %s -> %s", dst, trash)


def _read_checkpoint_iteration(args: Namespace) -> int | None:
    return read_trainer_checkpoint_iteration(args)


def _move_aside(dst: Path) -> Path:
    trash = dst.parent / f".trash_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    dst.rename(trash)
    return trash


def _snapshot_dir(checkpoint_root: Path, iteration: int) -> Path:
    return checkpoint_root / f"iter_{iteration:07d}" / "debug_events"
