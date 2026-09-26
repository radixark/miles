import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_STATE_FILENAME = "state.pt"


def save_simple_checkpoint(directory: Path, data: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(data, directory / _STATE_FILENAME)


def load_simple_checkpoint(directory: Path) -> Any:
    path = directory / _STATE_FILENAME
    if not path.exists():
        logger.warning(f"no dataset state under {path}: the dataset starts where a fresh run's would")
        return None

    logger.info(f"load metadata from {path}")
    return torch.load(path)
