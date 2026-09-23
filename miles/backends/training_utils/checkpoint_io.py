"""Checkpoint directories: written collectively, complete at their final path."""

# TODO: isolate shard-writing failures on the synchronous checkpoint path.

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group


def write_checkpoint_dir(
    path: str | Path,
    write_shards: Callable[[Path], None],
    metadata: dict | None = None,
    *,
    overwrite: bool = True,
) -> None:
    """Write collectively, then atomically point ``path`` at the completed version.

    All ranks must call. Readers may still hold an older version, so retain it.
    """
    publication = CheckpointPublication(path, metadata, overwrite=overwrite)
    publication.prepare()
    write_shards(publication.tmp_dir)
    _barrier()
    publication.publish()


class CheckpointIOError(OSError):
    """All ranks observed the same checkpoint directory IO failure."""


class CheckpointPublication:
    def __init__(self, path: str | Path, metadata: dict | None = None, *, overwrite: bool = True):
        self.final_dir = Path(path)
        self.tmp_dir = self.final_dir.parent / f"_tmp_{self.final_dir.name}"
        self.metadata = metadata
        self.overwrite = overwrite

    def prepare(self) -> None:
        self._on_rank_zero(self._prepare)

    def publish(self) -> None:
        self._on_rank_zero(self._publish)

    def _prepare(self):
        if not self.overwrite and self.final_dir.exists():
            raise FileExistsError(f"checkpoint {self.final_dir} already exists")
        if self.final_dir.exists() and not self.final_dir.is_symlink():
            raise NotImplementedError(
                f"cannot overwrite a legacy checkpoint directory {self.final_dir}; save under a new name"
            )
        # A crashed attempt may leave shards or an unpublished version link.
        if self.tmp_dir.is_symlink():
            self.tmp_dir.unlink()
        elif self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True)

    def _publish(self):
        if self.metadata is not None:
            (self.tmp_dir / "META.json").write_text(json.dumps(self.metadata, indent=2))
        version_dir = self.final_dir.parent / f"_version_{self.final_dir.name}_{uuid4().hex}"
        os.replace(self.tmp_dir, version_dir)
        self.tmp_dir.symlink_to(version_dir.name, target_is_directory=True)
        os.replace(self.tmp_dir, self.final_dir)

    @staticmethod
    def _on_rank_zero(operation):
        error = [None]
        if _rank() == 0:
            try:
                operation()
            except OSError as exc:
                error[0] = str(exc)
        if dist.is_initialized():
            dist.broadcast_object_list(error, src=0, group=get_gloo_group())
        if error[0] is not None:
            raise CheckpointIOError(error[0])


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
