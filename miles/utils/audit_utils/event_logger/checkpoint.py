"""Snapshot/restore the event directory alongside model checkpoints."""

import logging
import shutil
import time
import uuid
from argparse import Namespace
from pathlib import Path

logger = logging.getLogger(__name__)

_TRACKER_FILENAME = "latest_checkpointed_iteration.txt"


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

    iteration = _read_tracker_iteration(Path(args.load))
    if iteration is None:
        return

    src = _snapshot_dir(Path(args.load), iteration)
    if not src.is_dir():
        return

    dst = Path(args.save_debug_event_data)
    if dst.exists():
        trash = dst.parent / f".trash_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        dst.rename(trash)
        logger.info("Moved pre-restore event dir %s -> %s", dst, trash)
    shutil.copytree(src, dst)
    logger.info("Restored event dir %s <- %s", dst, src)


def _snapshot_dir(checkpoint_root: Path, iteration: int) -> Path:
    return checkpoint_root / f"iter_{iteration:07d}" / "debug_events"


def _read_tracker_iteration(checkpoint_root: Path) -> int | None:
    tracker = checkpoint_root / _TRACKER_FILENAME
    if not tracker.is_file():
        return None

    content = tracker.read_text().strip()
    if not content.isdigit():
        return None
    return int(content)
