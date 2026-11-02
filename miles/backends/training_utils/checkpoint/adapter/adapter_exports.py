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
    """Each module gets its own A, except that HF-fused components with one A go back to their HF module."""
    expanded = dict(weights)
    for group in parametrization.shared_lora_a_groups:
        shared_a = expanded.pop(f"{group[0]}.lora_A")
        for member in group:
            # safetensors refuses tensors that share storage
            expanded[f"{member}.lora_A"] = shared_a.clone()
    parents = {key.removesuffix(".lora_A").rpartition(".")[0] for key in expanded if key.endswith(".lora_A")}
    for parent in sorted(parents):
        for fused_leaf, leaves in fused_leaves(parent, targets).items():
            components = [f"{parent}.{leaf}" for leaf in leaves]
            if not all(f"{c}.lora_A" in expanded for c in components):
                continue
            a_by_component = [expanded[f"{c}.lora_A"] for c in components]
            if not all(torch.equal(a, a_by_component[0]) for a in a_by_component[1:]):
                assert fused_leaf != "gate_up_proj", f"{parent}: packed experts need gate_proj and up_proj to share A"
                continue
            fused = f"{parent}.{fused_leaf}"
            for c in components:
                expanded.pop(f"{c}.lora_A")
            expanded[f"{fused}.lora_A"] = a_by_component[0]
            expanded[f"{fused}.lora_B"] = fuse_lora_b(
                fused, [expanded.pop(f"{c}.lora_B") for c in components], targets
            )
    return expanded


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
    """bf16 PEFT with per-expert 2D tensors (packed families keep them stacked); a shared-outer side stays
    ``[1, ...]`` under the unindexed name."""
    tensors = {}
    for key, value in _expand_shared_lora_a(weights, parametrization, targets).items():
        value = value.to(torch.bfloat16).contiguous()
        per_expert = targets.packed_expert_gate_up_layout is None
        if per_expert and ".experts." in key and value.dim() == 3 and value.shape[0] > 1:
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
    for key, value in _expand_shared_lora_a(weights, parametrization, targets).items():
        name = map_hf_keys_to_peft_keys(key)
        for hf_fragment, sglang_fragment in SAMPLER_EXPERT_NAMES_BY_HF.items():
            name = name.replace(hf_fragment, sglang_fragment)
        tensors[name] = value.to(torch.bfloat16).contiguous()
    save_adapter_to_disk(directory, _adapter_config(parametrization, tensors), tensors)
    write_parametrization(Path(directory), parametrization)
