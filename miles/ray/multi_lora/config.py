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

    @property
    def serving_name(self) -> str:
        from miles.ray.multi_lora.identity import serving_lora_name

        return serving_lora_name(self.name, self.registration_id)


def parse_adapter_run_yaml(path: Path) -> AdapterRunConfig:
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {"rank", "save", "num_step", "metadata"}
    if unknown := set(raw) - known:
        raise ValueError(f"adapter yaml {path} has unsupported fields: {sorted(unknown)} (allowed: {sorted(known)})")
    return AdapterRunConfig(
        rank=raw.get("rank"),
        save=Path(raw["save"]) if raw.get("save") else None,
        num_step=raw.get("num_step"),
        metadata=raw.get("metadata") or {},
    )
