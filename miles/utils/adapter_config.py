"""Adapter config parsing for multi-LoRA training.

``AdapterConfig`` carries only static, YAML-sourced configuration; the
mutable slot is owned by the controller and exposed through ``RegisteredAdapter``
views.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AdapterConfig:

    # resolves them to CLI defaults if None (--lora-rank / --lora-alpha) on register.
    rank: int | None
    alpha: int | None

    data: str
    save: str | Path | None

    input_key: str
    label_key: str
    metadata_key: str | None = None

    rm_type: str | None = None
    custom_rm_path: str | None = None

    num_epoch: int | None = None
    num_row: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RegisteredAdapter:
    """Join view of an adapter's static config and current slot.

    Returned by the controller's ``active_adapters``. The controller is the
    source of truth; this view is a read-only snapshot.
    """

    name: str
    config: AdapterConfig
    slot: int


def parse_adapter_yaml(path: Path) -> AdapterConfig:
    """Parse a single adapter.yaml file.

    ``rank``, ``alpha`` and ``save`` are optional in the YAML; when absent the
    caller (e.g. the multi-LoRA controller) is responsible for resolving them.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    return AdapterConfig(
        rank=raw.get("rank"),
        alpha=raw.get("alpha"),
        data=raw["data"],
        save=Path(raw["save"]) if raw.get("save", None) else None,
        input_key=raw.get("input_key", "text"),
        label_key=raw.get("label_key"),
        metadata_key=raw.get("metadata_key"),
        rm_type=raw.get("rm_type"),
        custom_rm_path=raw.get("custom_rm_path"),
        num_epoch=raw.get("num_epoch"),
        num_row=raw.get("num_row"),
        metadata=raw.get("metadata") or {},
    )
