import logging
from pathlib import Path
from typing import Any

import torch

from miles.utils.file_utils import atomic_torch_save

logger = logging.getLogger(__name__)

_STATE_FILENAME = "state.pt"


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
