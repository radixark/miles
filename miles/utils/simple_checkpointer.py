import logging
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from miles.utils.file_utils import atomic_torch_save

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimpleCheckpointer:
    path_template: str
    require_exists: bool = False

    def save(self, args: Namespace, rollout_id: int, data: Any) -> None:
        if args.save is None:
            return
        path = self.path(args.save, rollout_id=rollout_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(path=path, obj=data)
        logger.info(f"Saved checkpoint to {path}")

    def load(self, args: Namespace, rollout_id: int | None) -> Any:
        if args.load is None:
            logger.warning("no --load: no checkpoint loaded")
            return None
        path = self.path(args.load, rollout_id=rollout_id)
        if not path.exists():
            if self.require_exists:
                raise FileNotFoundError(path)
            logger.warning(f"No checkpoint found at {path}")
            return None
        logger.info(f"Loading checkpoint from {path}")
        return torch.load(path, weights_only=False)

    def path(self, directory: str | Path, *, rollout_id: int | None) -> Path:
        return Path(directory) / self.path_template.format(rollout_id=rollout_id)
