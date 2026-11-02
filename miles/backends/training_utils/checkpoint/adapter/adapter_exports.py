"""PEFT and sampler exports, written from whole HF-keyed tensors on one process."""

from pathlib import Path

import torch

from miles.backends.training_utils.checkpoint.adapter.key_mapping import map_hf_keys_to_peft_keys
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    AdapterParametrization,
    write_parametrization,
)
from miles.utils.lora.hf_lora_targets import GDN_IN_PROJ_COMPONENTS_BY_MODULE
from miles.utils.lora.utils import get_adapter_target_modules, save_adapter_to_disk

_FUSED_MODULE_BY_COMPONENTS = {
    components: module for module, components in GDN_IN_PROJ_COMPONENTS_BY_MODULE.items() if len(components) > 1
}
# sglang's stacked-expert names (main's native export serves the same)
_SAMPLER_EXPERT_NAMES = {
    ".experts.gate_proj.": ".experts.w1.",
    ".experts.up_proj.": ".experts.w3.",
    ".experts.down_proj.": ".experts.w2.",
}


def _expand_shared_lora_a(weights: dict[str, torch.Tensor], parametrization: AdapterParametrization) -> dict:
    expanded = dict(weights)
    for group in parametrization.shared_lora_a_groups:
        leaves = tuple(m.rsplit(".", 1)[1] for m in group)
        fused = _FUSED_MODULE_BY_COMPONENTS.get(leaves)
        if fused is not None:
            # an HF-fused module whose components share A goes back to its HF name: one A, B stacked [q | k | v]
            parent = group[0].rsplit(".", 1)[0]
            expanded[f"{parent}.{fused}.lora_A"] = expanded.pop(f"{group[0]}.lora_A")
            expanded[f"{parent}.{fused}.lora_B"] = torch.cat([expanded.pop(f"{m}.lora_B") for m in group])
        else:
            for member in group[1:]:
                # safetensors refuses tensors that share storage
                expanded[f"{member}.lora_A"] = expanded[f"{group[0]}.lora_A"].clone()
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
    directory: Path, weights: dict[str, torch.Tensor], parametrization: AdapterParametrization
) -> None:
    """bf16 PEFT with per-expert 2D tensors; a shared-outer side stays ``[1, ...]`` under the unindexed name."""
    tensors = {}
    for key, value in _expand_shared_lora_a(weights, parametrization).items():
        value = value.to(torch.bfloat16).contiguous()
        if ".experts." in key and value.dim() == 3 and value.shape[0] > 1:
            for expert_index in range(value.shape[0]):
                tensors[map_hf_keys_to_peft_keys(key, expert_index)] = value[expert_index].clone()
        else:
            tensors[map_hf_keys_to_peft_keys(key)] = value
    save_adapter_to_disk(directory, _adapter_config(parametrization, tensors), tensors)


def write_sampler_adapter(
    directory: Path, weights: dict[str, torch.Tensor], parametrization: AdapterParametrization
) -> None:
    """bf16 adapter sglang loads from disk; experts stay stacked 3D. Carries its parametrization, so it reads back."""
    tensors = {}
    for key, value in _expand_shared_lora_a(weights, parametrization).items():
        name = map_hf_keys_to_peft_keys(key)
        for hf_fragment, sglang_fragment in _SAMPLER_EXPERT_NAMES.items():
            name = name.replace(hf_fragment, sglang_fragment)
        tensors[name] = value.to(torch.bfloat16).contiguous()
    save_adapter_to_disk(directory, _adapter_config(parametrization, tensors), tensors)
    write_parametrization(Path(directory), parametrization)
