"""Checkpoint directories: written collectively, complete at their final path."""

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group


def run_checkpoint_phase(step: Callable[[], None]) -> None:
    """Run one collective checkpoint step; any rank's failure raises on every rank."""
    error = None
    try:
        step()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    if not dist.is_initialized():
        if error is not None:
            raise RuntimeError(error)
        return
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error, group=get_gloo_group())
    failed = [e for e in errors if e is not None]
    if failed:
        raise RuntimeError(f"checkpoint phase failed on {len(failed)} rank(s): {failed[0]}")


def write_checkpoint_dir(path: str | Path, write_shards: Callable[[Path], None]) -> None:
    """Fill a fresh tmp dir through ``write_shards``, then move it to ``path``:
    a directory at its final path is always complete, and on overwrite the old
    version survives (as ``_old_<name>``) until the replacement is in place.
    Collective: every rank must call, whether or not it writes files."""
    final_dir = Path(path)
    tmp_dir = final_dir.parent / f"_tmp_{final_dir.name}"

    def make_tmp_dir():
        if _rank() == 0:
            if tmp_dir.exists():  # left over from a crashed attempt; stale shards must not join this write
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)

    def publish_dir():
        if _rank() != 0:
            return
        if final_dir.exists():
            old_dir = final_dir.parent / f"_old_{final_dir.name}"
            if old_dir.exists():
                shutil.rmtree(old_dir)
            os.replace(final_dir, old_dir)
            os.replace(tmp_dir, final_dir)
            shutil.rmtree(old_dir)
        else:
            os.replace(tmp_dir, final_dir)

    run_checkpoint_phase(make_tmp_dir)
    run_checkpoint_phase(lambda: write_shards(tmp_dir))
    run_checkpoint_phase(publish_dir)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0
