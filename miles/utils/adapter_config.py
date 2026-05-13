"""Adapter config parsing and lifecycle state for multi-LoRA training.

Each adapter directory contains an adapter.yaml. Lifecycle state is owned
by ``MultiLoRAController`` and snapshotted onto each ``AdapterConfig``.
"""

from enum import IntEnum, auto
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class AdapterState(IntEnum):
    """PENDING → ACTIVE → DRAINING → DRAINED → REMOVED."""

    PENDING = auto()             # registered, awaiting install
    ACTIVE = auto()              # installed, emitting samples
    DRAINING_DATASOURCE = auto() # data source will emit samples for at most one last train iteration
    DRAINING_INFLIGHT = auto()   # data source blocked, waiting for all in-flight requests to be drained
    DRAINING_TRAINABLE = auto()  # inflight samples complete, waiting for all trainable to be drained
    DRAINED = auto()             # all in-flight work trained; ready for cleanup

# Adapter in this state can generate samples during rollout
ADAPTER_ROLLOUT_STATES = {
    AdapterState.ACTIVE,
    AdapterState.DRAINING_DATASOURCE,
}
# Adapters in this state should not be trained on or have any generated samples
ADAPTER_INACTIVE_STATES = {
    AdapterState.PENDING,
    AdapterState.DRAINED
}

@dataclass(frozen=True)
class AdapterConfig:

    rank: int
    alpha: int

    # Path to data file
    data: str
    # Path to working directory for LoRA (checkpoints, artifacts, etc)
    dir: str | Path

    input_key: str
    label_key: str
    metadata_key: str | None = None

    rm_type: str | None = None
    custom_rm_path: str | None = None

    num_epoch: int | None = None
    num_row: int | None = None

    slot: int = -1
    state: AdapterState = AdapterState.PENDING

    def __post_init__(self):
        if not self.rm_type and not self.custom_rm_path:
            raise ValueError(f"Only one of rm_type or custom_rm_path should be set in AdapterConfig")
        if self.rm_type and self.custom_rm_path:
            raise ValueError(f"Only one of rm_type or custom_rm_path should be set in AdapterConfig")


def parse_adapter_yaml(path: Path) -> AdapterConfig:
    """Parse a single adapter.yaml file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    return AdapterConfig(
        rank=raw["rank"],
        alpha=raw["alpha"],
        data=raw["data"],
        dir=Path(raw["dir"]) if raw.get("dir", None) else path.parent,
        input_key=raw.get("input_key", "text"),
        label_key=raw.get("label_key"),
        metadata_key=raw.get("metadata_key"),
        rm_type=raw.get("rm_type"),
        custom_rm_path=raw.get("custom_rm_path"),
        num_epoch=raw.get("num_epoch"),
        num_row=raw.get("num_row"),
    )
