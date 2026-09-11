"""Snapshot/restore the event directory alongside model checkpoints."""

import logging
import shutil
import time
import uuid
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from miles.backends.megatron_utils.checkpoint_tracker import read_checkpoint_tracker_iteration
from miles.backends.megatron_utils.megatron_config import compute_trainer_checkpoint_dir, resolve_megatron_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventLoggerCheckpointer:
    require_exists: bool = False

    def snapshot(self, args: Namespace, iteration: int) -> None:
        if args.save_debug_event_data is None or args.save is None:
            return

        src = Path(args.save_debug_event_data)
        if not src.is_dir():
            if self.require_exists:
                raise FileNotFoundError(src)
            return

        dst = compute_event_snapshot_path(Path(args.save), iteration)
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            src, dst, ignore=shutil.ignore_patterns("sample_ownership_current", "sample_ownership_current.json")
        )
        logger.info("Snapshotted event dir %s -> %s", src, dst)

    def restore(self, args: Namespace, *, iteration: int | None = None) -> None:
        directory = args.requested_load if iteration is None else args.load
        if args.save_debug_event_data is None or directory is None:
            return

        requested_load = Path(directory)
        if iteration is None:
            iteration = _read_checkpoint_iteration(args)
        if iteration is None:
            return

        src = compute_event_snapshot_path(requested_load, iteration)
        if not src.is_dir():
            if self.require_exists:
                raise FileNotFoundError(src)
            return

        dst = Path(args.save_debug_event_data)
        if dst.exists():
            trash = _move_aside(dst)
            logger.info("Moved pre-restore event dir %s -> %s", dst, trash)
        shutil.copytree(src, dst)
        logger.info("Restored event dir %s <- %s", dst, src)


def snapshot(args: Namespace, iteration: int) -> None:
    EventLoggerCheckpointer().snapshot(args=args, iteration=iteration)


def restore(args: Namespace) -> None:
    EventLoggerCheckpointer().restore(args=args)


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
    leader = resolve_megatron_config(args).trainers[0]
    load_dir = (
        compute_trainer_checkpoint_dir(base_dir=args.requested_load, trainer_id=leader.trainer_id)
        if leader.model_id is not None
        else args.requested_load
    )
    return read_checkpoint_tracker_iteration(load_dir)


def _move_aside(dst: Path) -> Path:
    trash = dst.parent / f".trash_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    dst.rename(trash)
    return trash


def compute_event_snapshot_path(checkpoint_root: Path, iteration: int) -> Path:
    return checkpoint_root / f"iter_{iteration:07d}" / "debug_events"
