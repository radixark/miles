"""Checkpoint directory writes with completion metadata and failure cleanup."""

# TODO: isolate checkpoint IO failures in Tinker; they still terminate the trainer cell.

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import torch.distributed as dist

from miles.utils.distributed_utils import RANK_LOCAL_ERRORS, RankFailureError, run_on_all_ranks, run_on_rank0

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

    def prepare():
        if checkpoint_dir.exists():
            if not overwrite:
                raise FileExistsError(f"checkpoint {checkpoint_dir} already exists")
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True)

    def finalize():
        if metadata is not None:
            (checkpoint_dir / "META.json").write_text(json.dumps(metadata, indent=2))
        if completion_marker is not None:
            (checkpoint_dir / completion_marker).touch()

    run_on_rank0(f"preparing checkpoint {checkpoint_dir}", prepare)
    try:
        run_on_all_ranks(f"writing checkpoint {checkpoint_dir}", write_shards, checkpoint_dir)
        run_on_rank0(f"finalizing checkpoint {checkpoint_dir}", finalize)
    except RANK_LOCAL_ERRORS as error:
        # Delete only after the error exchange completed: then every rank has stopped writing. If the exchange
        # itself failed, peers may still be writing into the directory, so leave it (it has no completion marker).
        if is_rank0 and (isinstance(error, RankFailureError) or not distributed):
            try:
                shutil.rmtree(checkpoint_dir)
            except OSError:
                logger.exception(f"Failed to clean up checkpoint {checkpoint_dir}")
        raise
