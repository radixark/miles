"""Checkpoint directories: written collectively, complete at their final path."""

# TODO: isolate checkpoint IO failures; they currently terminate the trainer cell.

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group


def write_checkpoint_dir(
    path: str | Path,
    write_shards: Callable[[Path], None],
    metadata: dict | None = None,
    *,
    overwrite: bool = True,
) -> None:
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
        if not overwrite and final_dir.exists():
            raise FileExistsError(f"checkpoint {final_dir} already exists")
        if metadata is not None:
            (tmp_dir / "META.json").write_text(json.dumps(metadata, indent=2))
        if final_dir.exists():
            old_dir = final_dir.parent / f"_old_{final_dir.name}"
            if old_dir.exists():
                shutil.rmtree(old_dir)
            os.replace(final_dir, old_dir)
            try:
                os.replace(tmp_dir, final_dir)
            except OSError:
                os.replace(old_dir, final_dir)
                raise
            shutil.rmtree(old_dir)
        else:
            os.replace(tmp_dir, final_dir)

    make_tmp_dir()
    _barrier()
    write_shards(tmp_dir)
    _barrier()
    publish_dir()
    _barrier()


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
