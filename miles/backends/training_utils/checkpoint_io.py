"""Checkpoint directory writes with completion metadata and failure cleanup."""

# TODO: isolate checkpoint IO failures in Tinker; they still terminate the trainer cell.

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)


def write_checkpoint_dir(
    path: str | Path,
    write_shards: Callable[[Path], None],
    metadata: dict | None = None,
    *,
    overwrite: bool = True,
    completion_marker: str | None = None,
) -> None:
    """Replace a checkpoint; callers must exclude concurrent readers.

    All ranks must call and finish weight collectives before raising local write errors.
    """
    checkpoint_dir = Path(path)
    distributed = dist.is_initialized()
    is_rank0 = not distributed or dist.get_rank() == 0
    prepare_error = [None]
    if is_rank0:
        try:
            if checkpoint_dir.exists():
                if not overwrite:
                    raise FileExistsError(f"checkpoint {checkpoint_dir} already exists")
                shutil.rmtree(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True)
        except Exception as exc:
            prepare_error[0] = exc
    if distributed:
        dist.broadcast_object_list(prepare_error, src=0, group=get_gloo_group())
    if prepare_error[0] is not None:
        raise prepare_error[0]

    write_error = None
    try:
        write_shards(checkpoint_dir)
    except Exception as exc:
        write_error = exc
    errors = []
    if distributed:
        # This also waits for every writer; a failed collective must not trigger directory cleanup.
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(
            errors,
            f"{type(write_error).__name__}: {write_error}" if write_error is not None else None,
            group=get_gloo_group(),
        )
    try:
        if write_error is not None:
            raise write_error
        if any(errors):
            raise RuntimeError(
                "Checkpoint write failed: "
                + "; ".join(f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None)
            )
        if is_rank0:
            if metadata is not None:
                (checkpoint_dir / "META.json").write_text(json.dumps(metadata, indent=2))
            if completion_marker is not None:
                (checkpoint_dir / completion_marker).touch()
    except Exception:
        if is_rank0:
            try:
                shutil.rmtree(checkpoint_dir)
            except OSError:
                logger.exception(f"Failed to clean up checkpoint {checkpoint_dir}")
        raise
