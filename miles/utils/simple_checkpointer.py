import logging
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from miles.utils.file_utils import atomic_torch_save

logger = logging.getLogger(__name__)

_STATE_FILENAME = "state.pt"
_TEMPORARY_PREFIX = ".tmp-"


def save_simple_checkpoint(directory: Path, data: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _STATE_FILENAME
    atomic_torch_save(path=path, obj=data)
    logger.info(f"Saved checkpoint to {path}")


def load_simple_checkpoint(directory: Path) -> Any:
    path = directory / _STATE_FILENAME
    assert path.is_file(), f"{path} is missing, but the directory holding it is only ever published whole"
    logger.info(f"Loading checkpoint from {path}")
    return torch.load(path, weights_only=False)


@contextmanager
def atomic_save_folder(target: Path) -> Iterator[Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    for stale in target.parent.glob(f"{_TEMPORARY_PREFIX}*"):
        shutil.rmtree(stale)
        logger.info(f"Removed {stale}, left behind by a save that never finished")

    dir_temp = target.parent / f"{_TEMPORARY_PREFIX}{target.name}-{os.getpid()}"
    dir_temp.mkdir()
    try:
        yield dir_temp
    except Exception:
        shutil.rmtree(dir_temp, ignore_errors=True)
        raise

    replaced = target.parent / f"{_TEMPORARY_PREFIX}replaced-{target.name}-{os.getpid()}"
    if target.exists():
        os.replace(target, replaced)
    os.replace(dir_temp, target)
    if replaced.exists():
        shutil.rmtree(replaced)
