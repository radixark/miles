"""Snapshot/restore the event directory alongside model checkpoints."""

import logging
import shutil
import time
import uuid
from argparse import Namespace
from pathlib import Path

from miles.backends.megatron_utils.checkpoint_tracker import read_checkpoint_tracker_iteration

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
    if args.save_debug_event_data is None or args.load is None:
        return

    dst = Path(args.save_debug_event_data)
    iteration = read_checkpoint_tracker_iteration(Path(args.load))
    if iteration is None:
        _discard_abandoned_events(dst)
        return

    src = _snapshot_dir(Path(args.load), iteration)
    if not src.is_dir():
        return

    if dst.exists():
        trash = dst.parent / f".trash_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        dst.rename(trash)
        logger.info("Moved pre-restore event dir %s -> %s", dst, trash)
    shutil.copytree(src, dst)
    logger.info("Restored event dir %s <- %s", dst, src)


def _discard_abandoned_events(event_dir: Path) -> None:
    if event_dir.is_symlink():
        raise RuntimeError(f"Refusing to discard symbolic link event directory {event_dir}")
    if not event_dir.exists():
        return

    shutil.rmtree(event_dir)
    logger.info("Discarded pre-restart event dir %s because no checkpoint exists", event_dir)


def _snapshot_dir(checkpoint_root: Path, iteration: int) -> Path:
    return checkpoint_root / f"iter_{iteration:07d}" / "debug_events"
