"""Adapter config parsing and lifecycle state for multi-LoRA training.

``AdapterConfig`` carries only static, YAML-sourced configuration; the
mutable lifetime fields (slot, state) are owned by
``MultiLoRAController`` and exposed through ``RegisteredAdapter`` views.
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from typing import Any

import yaml


class AdapterState(IntEnum):
    """PENDING → RUNNING → DRAINING → DRAINED → REMOVED."""

    PENDING = auto()  # registered, awaiting install
    RUNNING = auto()  # installed on the model, emitting samples, eligible for training
    DRAINING_DATASOURCE = auto()  # data source will emit samples for at most one last train iteration
    DRAINING_INFLIGHT = auto()  # data source blocked, waiting for all in-flight requests to be drained
    DRAINING_TRAINABLE = auto()  # inflight samples complete, waiting for all trainable to be drained
    DRAINED = auto()  # all in-flight work trained; ready for cleanup
    REMOVED = auto()  # cross-system cleanup done; tombstone for external pollers


# Adapter in this state can generate samples during rollout
ADAPTER_ROLLOUT_STATES = {
    AdapterState.RUNNING,
    AdapterState.DRAINING_DATASOURCE,
}
# Adapters in this state should not be trained on or have any generated samples
ADAPTER_INACTIVE_STATES = {AdapterState.PENDING, AdapterState.DRAINED}


@dataclass(frozen=True)
class AdapterConfig:

    # rank/alpha may be None straight out of YAML; the multi-LoRA controller
    # resolves them to CLI defaults (--lora-rank / --lora-alpha) on register.
    rank: int | None
    alpha: int | None

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

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rm_type and not self.custom_rm_path:
            raise ValueError("Only one of rm_type or custom_rm_path should be set in AdapterConfig")
        if self.rm_type and self.custom_rm_path:
            raise ValueError("Only one of rm_type or custom_rm_path should be set in AdapterConfig")


@dataclass(frozen=True)
class RegisteredAdapter:
    """Join view of an adapter's static config and current lifetime state.

    Returned by ``MultiLoRAController.active_adapters``. The controller is the
    source of truth; this view is a read-only snapshot.
    """

    name: str
    config: AdapterConfig
    slot: int
    state: AdapterState


def parse_adapter_yaml(path: Path) -> AdapterConfig:
    """Parse a single adapter.yaml file.

    ``rank`` and ``alpha`` are optional in the YAML; when absent the caller
    (e.g. the multi-LoRA controller) is responsible for resolving them.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    return AdapterConfig(
        rank=raw.get("rank"),
        alpha=raw.get("alpha"),
        data=raw["data"],
        dir=Path(raw["dir"]) if raw.get("dir", None) else path.parent,
        input_key=raw.get("input_key", "text"),
        label_key=raw.get("label_key"),
        metadata_key=raw.get("metadata_key"),
        rm_type=raw.get("rm_type"),
        custom_rm_path=raw.get("custom_rm_path"),
        num_epoch=raw.get("num_epoch"),
        num_row=raw.get("num_row"),
        metadata=raw.get("metadata") or {},
    )
