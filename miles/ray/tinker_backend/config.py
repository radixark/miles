"""Registration config and read-only run views for the tinker backend.

A tinker training run is client-driven: no dataset, no reward, no server-side
batch shape. The public registration surface takes only ``rank`` (and
optional ``save``/``num_step``/``metadata``); ``alpha`` is server-resolved
from ``--lora-alpha`` and never client-settable."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdapterRunConfig:
    # LoRA rank; resolved against --lora-rank (ceiling) on register.
    rank: int | None = None
    # Server-internal: resolved from --lora-alpha; the public API never takes it.
    alpha: int | None = None
    # Checkpoint root; defaults to {--save}/adapters/{name}.
    save: str | Path | None = None
    # Optional client-set bound: auto-deregister after N optimizer steps.
    num_step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterRun:
    """Read-only join view of a run's static config and current clocks."""

    name: str
    config: AdapterRunConfig
    slot: int | None
    version: int = 0
    step: int = 0
    # Unique per registration: a re-registered name is a new tenant, and any
    # state stamped by the previous tenant must not carry over.
    registration_id: str = ""
