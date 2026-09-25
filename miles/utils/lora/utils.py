import json
import re
from argparse import Namespace
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

LORA_ADAPTER_NAME = "miles_lora"


def matches_lora_target(module: str, target: str) -> bool:
    return fnmatchcase(module if "." in target else module.rsplit(".", 1)[-1], target)


def is_lora_weight_name(name: str) -> bool:
    """Check if an HF weight name corresponds to a LoRA adapter weight."""
    return ".lora_A." in name or ".lora_B." in name


def is_lora_enabled(args: Namespace) -> bool:
    """Check if LoRA is enabled based on arguments."""
    return getattr(args, "lora_rank", 0) > 0 or getattr(args, "lora_adapter_path", None) is not None


def lora_resume_root(adapter_path: str | None) -> str | None:
    """The run root of an `<root>/iter_XXXXXXX/adapter` checkpoint, or None for any other adapter path."""
    if adapter_path is None:
        return None
    path = Path(adapter_path).resolve()
    if path.name != "adapter" or not re.fullmatch(r"iter_\d{7}", path.parent.name):
        return None
    return str(path.parent.parent)


def lora_rollout_enabled(args: Namespace) -> bool:
    """LoRA enabled AND the rollout side participates; false under --lora-train-only.

    Gates everything rollout-facing: SGLang's ``enable_lora``, the per-request
    ``lora_path``, and the adapter weight sync. Training-side LoRA is unaffected.
    """
    return is_lora_enabled(args) and not getattr(args, "lora_train_only", False)


def engine_loads_adapter_from_disk(args: Namespace) -> bool:
    """Only when no trainer will push the adapter; otherwise the first weight sync carries it."""
    return args.lora_adapter_path is not None and (args.debug_rollout_only or args.debug_skip_weight_update)


def lora_base_cpu_backup_enabled(args: Namespace) -> bool:
    """LoRA + --colocate + --lora-base-cpu-backup all set."""
    return is_lora_enabled(args) and getattr(args, "colocate", False) and getattr(args, "lora_base_cpu_backup", False)


def save_adapter_to_disk(out_dir, config: dict, tensors: dict) -> None:
    """Write a LoRA adapter dir (adapter_config.json + adapter_model.safetensors)."""
    import safetensors.torch  # lazy: this module is imported on paths that never touch weights

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    safetensors.torch.save_file(tensors, str(out / "adapter_model.safetensors"))


def build_lora_config(args, *, target_modules):
    return {
        "peft_type": "LORA",
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules if isinstance(target_modules, str) else list(target_modules),
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def get_adapter_target_modules(weight_names):
    return sorted({name.removeprefix("base_model.model.").rsplit(".lora_", 1)[0] for name in weight_names})


@dataclass(frozen=True)
class AdapterSpec:
    """The slot and scaling needed to export one adapter."""

    slot: int
    rank: int
    alpha: float


def is_multi_lora_enabled(args: Any) -> bool:
    return getattr(args, "multi_lora", False)


# Leaf module names that can live inside MoE experts (they also name the dense MLP
# projections); the bulk aliases expand to them during target-module resolution.
_EXPERT_LEAF_NAMES = frozenset({"linear_fc1", "linear_fc2", "gate_proj", "up_proj", "gate_up_proj", "down_proj"})
_ALL_MODULE_ALIASES = frozenset({"all", "all-linear", "all_linear"})


def targets_expert_leaves(target_modules: Any) -> bool:
    """Whether ``target_modules`` can put adapters on MoE expert linears."""
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    entries = [str(tm).strip().lower() for tm in (target_modules or [])]
    if any(entry in _ALL_MODULE_ALIASES for entry in entries):
        return True
    # Map each entry (possibly a dotted or wildcard path) to its leaf module name.
    return any(entry.split(".")[-1] in _EXPERT_LEAF_NAMES for entry in entries)
