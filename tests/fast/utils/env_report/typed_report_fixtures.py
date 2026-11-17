from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import model_serializer

from miles.utils.pydantic_utils import StrictBaseModel


class _SerializedReportConfig(StrictBaseModel):
    checkpoint: Path
    wandb_key: str

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, str]:
        return {"checkpoint": "serializer-mask", "wandb_key": self.wandb_key}


@dataclass
class _DataclassReportConfig:
    checkpoint: Path
    dtype: torch.dtype
    wandb_key: str
