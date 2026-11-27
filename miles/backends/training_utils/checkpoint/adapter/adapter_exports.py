"""PEFT, sampler and external (official Tinker) exports, written from whole HF-keyed tensors on one process."""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import safetensors.torch
import torch

from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    SAMPLER_EXPERT_NAMES_BY_HF,
    fuse_lora_b,
    fused_leaves,
    map_hf_keys_to_peft_export_keys,
    map_hf_keys_to_peft_keys,
    map_hf_keys_to_tinker_keys,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    AdapterParametrization,
    write_parametrization,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization_conversion import (
    apply_parametrization_conversion,
    plan_parametrization_conversion,
)
from miles.utils.lora.hf_lora_targets import HfLoraTargets
from miles.utils.lora.utils import get_adapter_target_modules, save_adapter_to_disk


def _components_without_plain_peft_form(parametrization: AdapterParametrization, targets: HfLoraTargets) -> list[str]:
    """Targeted components of an HF-fused module that do not share one A with every component of that module."""
    lonely = []
    for module in parametrization.target_modules:
        parent, _, leaf = module.rpartition(".")
        components = next((c for c in fused_leaves(parent, targets).values() if leaf in c), None)
        if components is None:
            continue
        members = {f"{parent}.{c}" for c in components}
        if not any(members <= set(group) for group in parametrization.shared_lora_a_groups):
            lonely.append(module)
    return sorted(lonely)


def has_plain_peft_form(parametrization: AdapterParametrization, targets: HfLoraTargets) -> bool:
    """Whether ``write_peft_adapter`` can name every tensor: HF has a module for a fused projection's components only
    when they share one A (e.g. canonical_lora's own A per GDN in_proj_q/k/v has none)."""
    return not _components_without_plain_peft_form(parametrization, targets)


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


def _adapter_config(parametrization: AdapterParametrization, tensor_names, lora_dropout: float) -> dict:
    assert parametrization.lora_alpha is not None, "lora_alpha must be resolved before an export is written"
    return {
        "peft_type": "LORA",
        "r": parametrization.lora_rank,
        "lora_alpha": parametrization.lora_alpha,
        "target_modules": get_adapter_target_modules(tensor_names),
        "lora_dropout": lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def write_peft_adapter(
    directory: Path,
    weights: dict[str, torch.Tensor],
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
    stacked_keys: frozenset[str] | None = None,
    *,
    lora_dropout: float,
) -> None:
    """bf16 PEFT with per-expert 2D tensors (experts re-packed into ``gate_up_proj`` stay stacked).

    ``stacked_keys``: the HF keys a backend's export named as one stacked tensor, which stay stacked; without it, the
    model family's packed layout decides.
    """
    # HF PEFT has no shared-outer form: every expert gets its own copy of the shared side
    per_expert = replace(parametrization, experts_shared_outer_loras=False)
    steps = plan_parametrization_conversion(parametrization, per_expert, source_dtype=torch.float32)
    weights = apply_parametrization_conversion(weights, steps)
    tensors = {}
    expanded = _expand_shared_lora_a(weights, per_expert, targets)
    lonely = _components_without_plain_peft_form(parametrization, targets)
    assert not lonely, (
        f"{lonely} are components of an HF-fused module with their own lora_A: this parametrization has no "
        "plain-PEFT form; the adapter dir is the checkpoint"
    )
    packed = _packed_experts(expanded)
    for key, value in expanded.items():
        value = value.to(torch.bfloat16).contiguous()
        stays_stacked = key in stacked_keys if stacked_keys is not None else _stays_stacked(key, packed, targets)
        if ".experts." in key and value.dim() == 3 and not stays_stacked:
            for expert_index in range(value.shape[0]):
                tensors[map_hf_keys_to_peft_export_keys(key, expert_index)] = value[expert_index].clone()
        else:
            tensors[map_hf_keys_to_peft_export_keys(key)] = value
    save_adapter_to_disk(directory, _adapter_config(parametrization, tensors, lora_dropout), tensors)


def write_sampler_adapter(
    directory: Path,
    weights: dict[str, torch.Tensor],
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    lora_dropout: float,
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
    # target_modules name the HF modules the adapter applies to, not sglang's tensor names for them
    hf_names = [map_hf_keys_to_peft_keys(key) for key in expanded]
    save_adapter_to_disk(directory, _adapter_config(parametrization, hf_names, lora_dropout), tensors)
    write_parametrization(Path(directory), parametrization)


def write_external_adapter(
    directory: Path,
    weights: dict[str, torch.Tensor],
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    lora_dropout: float,
    age_encryption_recipients: Sequence[str] = (),
) -> None:
    """The official Tinker export: F32 under Tinker names, experts stacked 3D as the parametrization has them
    (``[1, ...]`` on a shared-outer side), and every module its own copy of a shared A (no re-fusing: Tinker trains
    the components apart). With ``age_encryption_recipients`` both files are age-encrypted to them and named ``<file>.age``.
    """
    tensors = {}
    shared_copies = {member: group[0] for group in parametrization.shared_lora_a_groups for member in group[1:]}
    for module in parametrization.target_modules:
        for factor in ("lora_A", "lora_B"):
            source = f"{shared_copies.get(module, module)}.lora_A" if factor == "lora_A" else f"{module}.lora_B"
            tensor = weights[source].float()
            # safetensors refuses two names on one storage: only a shared A's later members get their own copy
            tensors[map_hf_keys_to_tinker_keys(f"{module}.{factor}", targets)] = (
                tensor.clone() if factor == "lora_A" and module in shared_copies else tensor
            )
    # as the official export: an HF leaf list would not match the Tinker-named experts
    config = _adapter_config(parametrization, (), lora_dropout) | {
        "target_modules": "all-linear",
        "modules_to_save": None,
        "target_parameters": None,
        "base_model_name_or_path": None,
    }
    if not age_encryption_recipients:
        save_adapter_to_disk(directory, config, tensors)
        return

    # lazy: every run imports this module, and prebuilt images may lack pyrage
    import pyrage

    # the SDK documents both native age (age1...) and SSH (ssh-...) public keys
    recipients = [
        (pyrage.ssh if r.startswith("ssh-") else pyrage.x25519).Recipient.from_str(r)
        for r in age_encryption_recipients
    ]
    # encrypted in memory, one file at a time: no plaintext ever reaches the disk
    Path(directory).mkdir(parents=True, exist_ok=True)
    for name, plaintext in (
        ("adapter_config.json", lambda: json.dumps(config, indent=2).encode()),
        ("adapter_model.safetensors", lambda: safetensors.torch.save(tensors)),
    ):
        (Path(directory) / f"{name}.age").write_bytes(pyrage.encrypt(plaintext(), recipients))
