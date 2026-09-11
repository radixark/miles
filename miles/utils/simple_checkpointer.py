import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class SimpleCheckpointer:
    path_template: str

    def save(self, path: str | Path, state_dict: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, path: str | Path) -> Any:
        state_dict = torch.load(path)
        return state_dict

    def path(self, directory: str | Path, *, rollout_id: int | None) -> str:
        return os.path.join(directory, self.path_template.format(rollout_id=rollout_id))
