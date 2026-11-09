"""The actor's published weight version as a torchtitan checkpoint state; imports no torchtitan."""

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful

PUBLISHED_WEIGHT_VERSION = "published_weight_version"


class PublishedWeightVersionState(Stateful):
    def __init__(self, actor):
        self._actor = actor

    def state_dict(self) -> dict:
        return {"version": self._actor._published_weight_version}

    def load_state_dict(self, state_dict: dict) -> None:
        self._actor._published_weight_version = state_dict["version"]


def saved_in(checkpoint_id: str) -> bool:
    saved_keys = dcp.FileSystemReader(checkpoint_id).read_metadata().state_dict_metadata
    return any(key.startswith(f"{PUBLISHED_WEIGHT_VERSION}.") for key in saved_keys)
