import json
import re
from argparse import Namespace
from fnmatch import fnmatchcase
from pathlib import Path

LORA_ADAPTER_NAME = "miles_lora"


def is_lora_weight_name(name: str) -> bool:
    """Check if an HF weight name corresponds to a LoRA adapter weight."""
    return ".lora_A." in name or ".lora_B." in name


def is_lora_enabled(args: Namespace) -> bool:
    """Check if LoRA is enabled based on arguments."""
    return getattr(args, "lora_rank", 0) > 0 or getattr(args, "lora_adapter_path", None) is not None


def lora_rollout_enabled(args: Namespace) -> bool:
    """LoRA enabled AND the rollout side participates; false under --lora-train-only.

    Gates everything rollout-facing: SGLang's ``enable_lora``, the per-request
    ``lora_path``, and the adapter weight sync. Training-side LoRA is unaffected.
    """
    return is_lora_enabled(args) and not getattr(args, "lora_train_only", False)


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
        "target_modules": list(target_modules),
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def _split_adapter_weight_name(name):
    module, separator, factor = name.removeprefix("base_model.model.").rpartition(".lora_")
    assert separator and factor in ("A.weight", "B.weight"), f"Unexpected adapter weight name: {name}"
    return module, factor[0]


def get_adapter_target_modules(weight_names):
    return sorted({_split_adapter_weight_name(name)[0] for name in weight_names})


def validate_adapter_export(weight_names, targets, *, shared_outer=False):
    factors = {"A": set(), "B": set()}
    for name in weight_names:
        module, factor = _split_adapter_weight_name(name)
        if shared_outer:
            # Shared-outer factors omit the expert index on exactly one side.
            module = re.sub(r"\bexperts\.\d+\.", "experts.", module)
        factors[factor].add(module)
    assert factors["A"], "Adapter export contains no LoRA weights"
    assert (
        factors["A"] == factors["B"]
    ), f"Adapter export has unpaired A/B modules: {sorted(factors['A'] ^ factors['B'])}"
    if shared_outer:
        targets = [target.replace(".experts.*.", ".experts.") for target in targets]
    modules = factors["A"]
    unexpected = {module for module in modules if not any(fnmatchcase(module, target) for target in targets)}
    missing = {target for target in targets if not any(fnmatchcase(module, target) for module in modules)}
    assert not unexpected, f"Adapter export includes unselected HF modules: {sorted(unexpected)}"
    assert not missing, f"Adapter export is missing selected HF targets: {sorted(missing)}"
