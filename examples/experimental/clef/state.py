"""Torch-only state helpers for supervised schema-head training.

Keep the standalone trainer independent of Ray actors and rollout dependencies.
The state layout matches the FSDP backend's native checkpoint wrappers.
"""

from typing import Any

import torch
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful


class ModelState(Stateful):
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        model_state, _ = get_state_dict(self.model, optimizers=[])
        return {"model": model_state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        set_state_dict(self.model, optimizers=[], model_state_dict=state["model"], optim_state_dict=None)


class OptimizerState(Stateful):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        _, optimizer_state = get_state_dict(self.model, optimizers=self.optimizer)
        return {"optim": optimizer_state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        set_state_dict(self.model, optimizers=self.optimizer, model_state_dict=None, optim_state_dict=state["optim"])


def apply_fp32_master(model: torch.nn.Module) -> torch.nn.Module:
    sync_dtypes = {name: value.dtype for name, value in model.state_dict().items()}
    model = model.to(torch.float32)
    model._fsdp_sync_dtypes = sync_dtypes
    return model
