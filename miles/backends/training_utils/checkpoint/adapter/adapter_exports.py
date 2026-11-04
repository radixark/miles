"""PEFT and sampler exports, written from whole HF-keyed tensors on one process."""

from dataclasses import replace
from pathlib import Path

import torch

from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    SAMPLER_EXPERT_NAMES_BY_HF,
    map_hf_keys_to_peft_export_keys,
    map_hf_keys_to_peft_keys,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    AdapterParametrization,
    write_parametrization,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization_conversion import (
    apply_parametrization_conversion,
    plan_parametrization_conversion,
)
from miles.utils.lora.hf_lora_targets import GDN_IN_PROJ_COMPONENTS_BY_MODULE
from miles.utils.lora.utils import get_adapter_target_modules, save_adapter_to_disk

_FUSED_MODULE_BY_COMPONENTS = {
    components: module for module, components in GDN_IN_PROJ_COMPONENTS_BY_MODULE.items() if len(components) > 1
}


def _components_without_plain_peft_form(parametrization: AdapterParametrization) -> list[str]:
    """Targeted components of an HF-fused module that do not share one A with every component of that module."""
    lonely = []
    for module in parametrization.target_modules:
        parent, _, leaf = module.rpartition(".")
        components = next((c for c in _FUSED_MODULE_BY_COMPONENTS if leaf in c), None)
        if components is None:
            continue
        members = {f"{parent}.{c}" for c in components}
        if not any(members <= set(group) for group in parametrization.shared_lora_a_groups):
            lonely.append(module)
    return sorted(lonely)


def has_plain_peft_form(parametrization: AdapterParametrization) -> bool:
    """Whether ``write_peft_adapter`` can name every tensor: HF has a module for a fused projection's components only
    when they share one A (e.g. canonical_lora's own A per GDN in_proj_q/k/v has none)."""
    return not _components_without_plain_peft_form(parametrization)


def _expand_shared_lora_a(weights: dict[str, torch.Tensor], parametrization: AdapterParametrization) -> dict:
    expanded = dict(weights)
    for group in parametrization.shared_lora_a_groups:
        shared_a = expanded.pop(f"{group[0]}.lora_A")
        parent = group[0].rpartition(".")[0]
        unfused = list(group)
        for components, fused in _FUSED_MODULE_BY_COMPONENTS.items():
            members = [f"{parent}.{c}" for c in components]
            if all(m in unfused for m in members):
                # an HF-fused module whose components share A goes back to its HF name: one A, B stacked [q | k | v]
                expanded[f"{parent}.{fused}.lora_A"] = shared_a.clone()
                expanded[f"{parent}.{fused}.lora_B"] = torch.cat([expanded.pop(f"{m}.lora_B") for m in members])
                unfused = [m for m in unfused if m not in members]
        for member in unfused:
            # safetensors refuses tensors that share storage
            expanded[f"{member}.lora_A"] = shared_a.clone()
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
    """bf16 PEFT with per-expert 2D tensors."""
    # HF PEFT has no shared-outer form: every expert gets its own copy of the shared side
    per_expert = replace(parametrization, experts_shared_outer_loras=False)
    steps = plan_parametrization_conversion(parametrization, per_expert, source_dtype=torch.float32)
    weights = apply_parametrization_conversion(weights, steps)
    expanded = _expand_shared_lora_a(weights, per_expert)
    lonely = _components_without_plain_peft_form(parametrization)
    assert not lonely, (
        f"{lonely} are components of an HF-fused module with their own lora_A: this parametrization has no "
        "plain-PEFT form; the adapter dir is the checkpoint"
    )
    tensors = {}
    for key, value in expanded.items():
        value = value.to(torch.bfloat16).contiguous()
        if ".experts." in key and value.dim() == 3:
            for expert_index in range(value.shape[0]):
                tensors[map_hf_keys_to_peft_export_keys(key, expert_index)] = value[expert_index].clone()
        else:
            tensors[map_hf_keys_to_peft_export_keys(key)] = value
    save_adapter_to_disk(directory, _adapter_config(parametrization, tensors), tensors)


def write_sampler_adapter(
    directory: Path, weights: dict[str, torch.Tensor], parametrization: AdapterParametrization
) -> None:
    """bf16 adapter sglang loads from disk; experts stay stacked 3D. Carries its parametrization, so it reads back."""
    tensors = {}
    for key, value in _expand_shared_lora_a(weights, parametrization).items():
        name = map_hf_keys_to_peft_keys(key)
        for hf_fragment, sglang_fragment in SAMPLER_EXPERT_NAMES_BY_HF.items():
            name = name.replace(hf_fragment, sglang_fragment)
        tensors[name] = value.to(torch.bfloat16).contiguous()
    save_adapter_to_disk(directory, _adapter_config(parametrization, tensors), tensors)
    write_parametrization(Path(directory), parametrization)
