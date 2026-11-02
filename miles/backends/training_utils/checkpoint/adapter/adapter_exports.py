"""PEFT and sampler exports, written from whole HF-keyed tensors on one process."""

from pathlib import Path

import torch

from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    SAMPLER_EXPERT_NAMES_BY_HF,
    fuse_lora_b,
    fused_leaves,
    map_hf_keys_to_peft_keys,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    AdapterParametrization,
    write_parametrization,
)
from miles.utils.lora.hf_lora_targets import HfLoraTargets
from miles.utils.lora.utils import get_adapter_target_modules, save_adapter_to_disk


def _expand_shared_lora_a(
    weights: dict[str, torch.Tensor], parametrization: AdapterParametrization, targets: HfLoraTargets
) -> dict:
    """Each module gets its own A, except that an HF-fused module whose components share A goes back to its name."""
    expanded = dict(weights)
    for group in parametrization.shared_lora_a_groups:
        shared_a = expanded.pop(f"{group[0]}.lora_A")
        parent = group[0].rpartition(".")[0]
        unfused = list(group)
        for fused_leaf, leaves in fused_leaves(parent, targets).items():
            members = [f"{parent}.{leaf}" for leaf in leaves]
            if all(m in unfused for m in members):
                fused = f"{parent}.{fused_leaf}"
                expanded[f"{fused}.lora_A"] = shared_a.clone()
                expanded[f"{fused}.lora_B"] = fuse_lora_b(
                    fused, [expanded.pop(f"{m}.lora_B") for m in members], targets
                )
                unfused = [m for m in unfused if m not in members]
        for member in unfused:
            # safetensors refuses tensors that share storage
            expanded[f"{member}.lora_A"] = shared_a.clone()
    return expanded


def _packed_experts(expanded: dict[str, torch.Tensor]) -> set[str]:
    """The ``.experts`` modules whose gate / up went back to one packed ``gate_up_proj``; they stay stacked 3D."""
    return {
        key.removesuffix(".gate_up_proj.lora_A") for key in expanded if key.endswith(".experts.gate_up_proj.lora_A")
    }


def _stays_stacked(key: str, packed: set[str], targets: HfLoraTargets) -> bool:
    # a packed family's HF down_proj is one 3D tensor whether or not gate_up_proj is adapted
    is_packed_down = targets.packed_expert_gate_up_layout is not None and ".experts.down_proj." in f"{key}."
    return _experts_module(key) in packed or is_packed_down


def _experts_module(key: str) -> str:
    return key.rsplit(".", 2)[0]


def _adapter_config(parametrization: AdapterParametrization, tensor_names) -> dict:
    assert parametrization.lora_alpha is not None, "lora_alpha must be resolved before an export is written"
    return {
        "peft_type": "LORA",
        "r": parametrization.lora_rank,
        "lora_alpha": parametrization.lora_alpha,
        "target_modules": get_adapter_target_modules(tensor_names),
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def write_peft_adapter(
    directory: Path, weights: dict[str, torch.Tensor], parametrization: AdapterParametrization, targets: HfLoraTargets
) -> None:
    """bf16 PEFT with per-expert 2D tensors (experts re-packed into ``gate_up_proj`` stay stacked); a shared-outer
    side stays ``[1, ...]`` under the unindexed name."""
    tensors = {}
    expanded = _expand_shared_lora_a(weights, parametrization, targets)
    packed = _packed_experts(expanded)
    for key, value in expanded.items():
        value = value.to(torch.bfloat16).contiguous()
        if ".experts." in key and value.dim() == 3 and value.shape[0] > 1 and not _stays_stacked(key, packed, targets):
            for expert_index in range(value.shape[0]):
                tensors[map_hf_keys_to_peft_keys(key, expert_index)] = value[expert_index].clone()
        else:
            tensors[map_hf_keys_to_peft_keys(key)] = value
    save_adapter_to_disk(directory, _adapter_config(parametrization, tensors), tensors)


def write_sampler_adapter(
    directory: Path, weights: dict[str, torch.Tensor], parametrization: AdapterParametrization, targets: HfLoraTargets
) -> None:
    """bf16 adapter sglang loads from disk; experts stay stacked 3D. Carries its parametrization, so it reads back."""
    tensors = {}
    expanded = _expand_shared_lora_a(weights, parametrization, targets)
    packed = _packed_experts(expanded)
    for key, value in expanded.items():
        name = map_hf_keys_to_peft_keys(key)
        # packed experts keep their HF names, as main's export serves them
        if not _stays_stacked(key, packed, targets):
            for hf_fragment, sglang_fragment in SAMPLER_EXPERT_NAMES_BY_HF.items():
                name = name.replace(hf_fragment, sglang_fragment)
        tensors[name] = value.to(torch.bfloat16).contiguous()
    save_adapter_to_disk(directory, _adapter_config(parametrization, tensors), tensors)
    write_parametrization(Path(directory), parametrization)
